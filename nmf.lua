require 'audio'
require 'image'

local fftw3 = require 'fftw3'
signal = require 'signal'

local function typecheck(input)
   if input:type() == 'torch.FloatTensor' then
      fftw = fftw3.float
      fftw_complex_cast = 'fftwf_complex*'
   elseif input:type() == 'torch.DoubleTensor' then
      fftw = fftw3
      fftw_complex_cast = 'fftw_complex*'
   else
      error('Unsupported precision of input tensor: '
               .. input:type()
               .. '  . Supported precision is Float/Double.')
   end
end

local function apply_window(window, window_type) -- from "signal"
   local window_type = window_type or 'rect'
   local window_size = window:size(1)
   local m = window_size - 1
   local window = window:contiguous()
   local wdata = torch.data(window)
   if window_type == 'hamming' then
      for i=0,window_size-1 do
         wdata[i] = wdata[i] * (.53836 - .46164 * math.cos(2 * math.pi * i / m));
      end
   elseif window_type == 'hann' then
      for i=0,window_size-1 do
         wdata[i] = wdata[i] * (.5 - .5 * math.cos(2 * math.pi * i / m));
      end
   elseif window_type == 'bartlett' then
      for i=0,window_size-1 do
         wdata[i] = wdata[i] * (2 / m * ((m/2) - math.abs(i - (m/2))));
      end
   end
   return window
end

function create_window(window_size, window_type)
   return apply_window(torch.ones(window_size), window_type)
end

function hanning(n)
   local m = n - 1
   local function h(k)
      return 0.5 - 0.5 * math.cos(2.0 * math.pi * k / m)
   end
   return torch.range(0, m):apply(h)
end


function rstft(input, window_size, window_stride, window_type)
   typecheck(input)
   if input:dim() ~= 1 then error('Need 1D Tensor input') end
   local length = input:size(1)
   local nwindows = math.floor(((length - window_size)/window_stride) + 1);
   local noutput  = math.floor(window_size/2 + 1);
   local output   = torch.Tensor(nwindows, noutput, 2):typeAs(input):zero()
   local window_index = 1
   local w = create_window(window_size, window_type)
   for i=1,length,window_stride do
      if (i+window_size-1) > length then break; end
      local window = input[{{i,i+window_size-1}}]
      -- apply preprocessing
      window:cmul(w)
      -- fft
      local winout = signal.rfft(window)
      output[window_index] = winout
      window_index = window_index + 1
   end
   return output
end

function irstft(input, window_size, window_stride, window_type)
   typecheck(input)
   if input:dim() ~= 3 or input:size(3) ~= 2 then
      error('Input has to be 3D Tensor of size NxMx2 (Complex input with NxM points)')
   end
   local noutput = input:size(1)
   local length = window_size + (noutput - 1) * window_stride
   local output = torch.Tensor(length):typeAs(input):zero()
   local windowsum = torch.Tensor(length):typeAs(input):zero()
   local w = create_window(window_size, window_type)
   local windowsquare = torch.pow(w, 2)
   -- inverse stft & window
   for i=1,noutput do
      local r = signal.irfft(input[i], window_size)
      r:cmul(w)
      local s = (i-1)*window_stride+1
      local e = (i-1)*window_stride+window_size
      local ts = {{s,e}}
      output[ts] = output[ts] + r
      windowsum[ts] = windowsum[ts] + windowsquare
   end
   -- normalize without zero-division
   -- for i=1,length do
   --    if windowsum[i] > 1.0e-7 then
   --       output[i] = output[i] / windowsum[i]
   --    end
   -- end
   local eps = 1.0e-7
   local mask = windowsum:lt(eps)
   windowsum:maskedFill(mask, 1.0)
   output:cdiv(windowsum)
   return output
end

function gain_phase(x)
   function sq(x) return torch.pow(x,2) end
   local gain = (sq(x:select(3,1)) + sq(x:select(3,2))):sqrt()
   local phase = torch.cdiv(x:select(3,1), x:select(3,2)):atan()
   return gain, phase
end

function nmf_euc(Y, K, maxiter)
   local K = K or 4
   local maxiter = maxiter or 100
   local eps = 1.0e-21
   local H = torch.rand(Y:size(1), K)
   local U = torch.rand(K, Y:size(2))
   for i = 1, maxiter do
      H:cmul(H, (Y * U:t()):cdiv(H * U * U:t() + eps))
      U:cmul(U, (H:t() * Y):cdiv(H:t() * H * U + eps))
      U = U / torch.max(U)
   end
   return H, U
end


xs, fs = audio.load "test10k.wav"
x = xs:select(2,1)
assert(fs == 10000)
framelen = 512
hopsize = math.floor(framelen / 2)
window_type = "hann"
X = rstft(x, framelen, hopsize, window_type)
rx = irstft(X, framelen, hopsize, window_type)
-- audio.save("rx.wav", rx:reshape(rx:nElement(), 1), fs)

Y, phase = gain_phase(X)
K = 4
torch.manualSeed(98765)
H, U = nmf_euc(Y)

chu = torch.Tensor(H:size(1),U:size(2),2)
c, s = torch.cos(phase), torch.sin(phase)
max_gain = torch.max(x)

for k=1,K do
   local hu = torch.ger(H[{{},k}], U[k])
   chu[{{},{},1}] = torch.cmul(hu, c)
   chu[{{},{},2}] = torch.cmul(hu, s)
   local y = irstft(chu, framelen, hopsize, window_type)
   audio.save(k .. "lua.wav", y:reshape(y:nElement(), 1), fs)
end


require 'gnuplot'
-- make this coroutine
fig_id = 0
function add_figure(title)
-- gnuplot.figure(fig_id)
-- gnuplot.title(title)
-- fig_id = fig_id + 1
gnuplot.pngfigure(title .. ".png")
end


add_figure("spectogram")
gnuplot.imagesc(torch.log(Y):t(), "color")
gnuplot.plotflush()

add_figure("freq-factor") -- todo log scale
gnuplot.plot(
    {U[1]},
    {U[2]},
    {U[3]},
    {U[4]}
)
gnuplot.plotflush()

add_figure("time-factor")
Ht = H:t()
gnuplot.plot(
    {Ht[1]},
    {Ht[2]},
    {Ht[3]},
    {Ht[4]}
)
gnuplot.plotflush()

