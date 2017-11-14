
--- Demo: Korean Syllable Generation  
-- 2017.11.13
-- Hwaran Lee @ KAIST 

require 'torch'
require 'nn'
require 'model.LSTM'
utf8 = require 'lua-utf8'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Korean sentence generation (syllable)')
cmd:text('Options')
cmd:option('-model_dir','exp/lm-rnn2048x2-v2-wd0-dw1/lm_char_epoch43.00.t7','trained model directory')
cmd:option('-vocab', 'exp/vocab.t7', 'model checkpoint file')


-- GPU/CPU these params must be passed in because it affects the constructors
cmd:option('-gpuid', 0,'which gpu to use. -1 = use CPU')
cmd:option('-cudnn', 0,'use cudnn (1 = yes, 0 = no)')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- global constants for certain tokens
tokens = {}
tokens.EOS = '<eos>' --<eos>
tokens.UNK = '<unk>' --<unk>
tokens.SOS = '<sos>' --<sos>'
tokens.BLK = '|'

-- load necessary packages depending on config options
dtype = 'torch.FloatTensor'
if opt.gpuid >= 0 then
  print('using CUDA on GPU ' .. opt.gpuid .. '...')
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpuid + 1)
  dtype = 'torch.CudaTensor'
end

--- vocabulary loading ---
print('--- Loading vocab : ' .. opt.vocab)
vocab = torch.load(opt.vocab)
idx2word, word2idx, _, _ = table.unpack(vocab)
print(string.format('Word vocab size: %d', #idx2word))

--- network loading ---
function resetStates_LSTM(protos)
  for i, v in pairs(protos.net:findModules('nn.LSTM')) do
    v:resetStates()
  end
end
function rememberStates_LSTM(protos)
  for i, v in pairs(protos.net:findModules('nn.LSTM')) do
    v.remember_states = true
  end
end

print('-- Building network --')
checkpoint = torch.load(opt.model_dir)
opt = checkpoint.opt
protos = checkpoint.protos
rememberStates_LSTM(protos)
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end


function txt2tensor(line)
  local word_num = 0
  for word in utf8.gmatch(line, '.') do
    word_num = word_num + 1
  end
  local x = torch.Tensor(word_num+1):long()
  x[1] = word2idx[tokens.SOS]
  
  word_num = 1
  for word in utf8.gmatch(line, '.') do
     word_num = word_num + 1
    if word == ' ' then word = tokens.BLK end
    if word2idx[word]==nil then
      x[word_num] = word2idx[tokens.UNK]
    else
      x[word_num] = word2idx[word]
    end        
  end
  return x:resize(1,word_num)
end

function tensor2txt(seq)
  line = ''
  for t =1, seq:size(1) do
    if idx2word[seq[t]] == tokens.UNK then break end    
    if idx2word[seq[t]] == tokens.BLK then
      line = line .. ' '
    else
      line = line .. idx2word[seq[t]]
    end
  end
  return line
end

function generation(preSeq, maxLen)
  -- preSeq is tensor
  local batch_size = 2
  
  preSeq = torch.cat(preSeq, preSeq, 1)
  preLen = preSeq:size(2)
  
  seq = torch.LongTensor(2,maxLen):fill(1)
  if opt.gpuid >= 0 then seq=seq:cuda() preSeq=preSeq:cuda() end

  resetStates_LSTM(protos)
  protos.net:evaluate()

  -- (1) forward the preSeq
  protos.view_in:resetSize(batch_size*preLen, -1)
  protos.view_out:resetSize(batch_size, preLen, -1)
  out = protos.net:forward(preSeq)  
  
  local function select_next(top) 
    i=1
    while i < #idx2word do
      if idx2word[top[i]] ~= tokens.UNK and idx2word[top[i]] ~= tokens.SOS then
        break
      end
      i = i+1
    end
    return top[i]
  end  

  _, top = torch.sort(out[{1,preLen}], 1, true)
  next_tok = select_next(top)
  seq:select(2,1):fill(top[1])
  if next_tok == tokens.EOS then return seq[{1,{}}] end
  seqLen = 1
  
  -- (2) generate seq iteratively
  for t = 1, maxLen-1 do
    protos.view_in:resetSize(batch_size*1, -1)
    protos.view_out:resetSize(batch_size, 1, -1)
    out = protos.net:forward(seq[{{},t}]:resize(2,1))
    _, top = torch.sort(out[{1,1,{}}], 1, true)
    next_tok = select_next(top)
    seq:select(2,t+1):fill(top[1])
    seqLen = seqLen+1
    if next_tok == word2idx[tokens.EOS] then break end
  end
  return seq[{1, {1, seqLen}}]
  
end

--- main --- 
while(true)
do
  print('======================================')

  io.write("Please enter a front part of sentence : \n")
  io.flush()
  input = io.read()
  if input == 'exit' then break end
  
  maxLen = 100
  preSeq = txt2tensor(input)
  outseq = generation(preSeq, maxLen)
  output = tensor2txt(outseq)
  output = input .. output
  print(' \n')
  print(output)
  print(' \n')
end 
