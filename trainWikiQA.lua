--[[
  Training script for semantic relatedness prediction on the Twitter dataset.
  We Thank Kai Sheng Tai for providing the preprocessing/basis codes. 
--]]

require('torch')
require('nn')
require('nngraph')
require('optim')
require('xlua')
require('sys')
require('lfs')
--require('../')
--require('./util')
--nngraph.setDebug(true)

similarityMeasure = {}

include('util/read_data.lua')
include('util/Vocab.lua')
include('CsDis.lua')
include('metric2.lua')
include('init.lua')

printf = utils.printf

-- global paths (modify if desired)
similarityMeasure.data_dir        = 'data'
similarityMeasure.models_dir      = 'trained_models'
similarityMeasure.predictions_dir = 'predictions'

function header(s)
  print(string.rep('-', 80))
  print(s)
  print(string.rep('-', 80))
end

-- read command line arguments
local args = lapp [[
Training script for semantic relatedness prediction on the Twitter dataset.
  -m,--model  (default dependency) Model architecture: [dependency, lstm, bilstm]
  -l,--layers (default 1)          Number of layers (ignored for Tree-LSTM)
  -d,--dim    (default 150)        LSTM memory dimension
]]

--local model_name, model_class, model_structure
--model_name = 'NTM'
--model_class = similarityMeasure.Conv
--model_structure = model_name

--torch.seed()
torch.manualSeed(123)
print('<torch> using the automatic seed: ' .. torch.initialSeed())

-- directory containing dataset files
local data_dir = 'data/WikiQA/'

-- load vocab
local vocab = similarityMeasure.Vocab(data_dir .. 'vocab.txt')

-- load embeddings
print('loading word embeddings')

--local emb_dir = '/scratch0/huah/textSimilarityConvNet/data/glove/'
--local emb_prefix = emb_dir .. 'glove.840B'
local emb_dir = '/scratch0/huah/textSimilarityConvNet/data/paragram_sl999_PPXXL/'
local emb_prefix = emb_dir .. 'paragram_sl999_PPXXL'

local emb_vocab, emb_vecs = similarityMeasure.read_embedding(emb_prefix .. '.vocab', emb_prefix .. '.300.th')

local emb_dim = emb_vecs:size(2)

-- use only vectors in vocabulary (not necessary, but gives faster training)
local num_unk = 0
local vecs = torch.Tensor(vocab.size, emb_dim)
for i = 1, vocab.size do
  local w = vocab:token(i)
  if emb_vocab:contains(w) then
    vecs[i] = emb_vecs[emb_vocab:index(w)]
  elseif i == vocab.size then
    vecs[i]:zero()
  else
    num_unk = num_unk + 1
    vecs[i]:uniform(-0.05, 0.05)
  end
end
print('unk count = ' .. num_unk)
emb_vocab = nil
emb_vecs = nil
collectgarbage()
taskD = 'qa'
-- load datasets
print('loading' .. taskD .. ' datasets')
local train_dir = data_dir .. 'pre-train/'
local dev_dir = data_dir .. 'dev/'
local test_dir = data_dir .. 'test/'
local train_dataset = similarityMeasure.read_relatedness_dataset(train_dir, vocab, taskD)
local dev_dataset = similarityMeasure.read_relatedness_dataset(dev_dir, vocab, taskD)
local test_dataset = similarityMeasure.read_relatedness_dataset(test_dir, vocab, taskD)

local lmax = math.max(train_dataset.lmaxsize, dev_dataset.lmaxsize, test_dataset.lmaxsize)
local rmax = math.max(train_dataset.rmaxsize, dev_dataset.rmaxsize, test_dataset.rmaxsize)

train_dataset.lmaxsize = lmax
dev_dataset.lmaxsize = lmax
test_dataset.lmaxsize = lmax
train_dataset.rmaxsize = rmax
dev_dataset.rmaxsize = rmax
test_dataset.rmaxsize = rmax
printf('lmax = %d | train lmax = %d | dev lmax = %d\n', lmax, train_dataset.lmaxsize, dev_dataset.lmaxsize)
printf('rmax = %d | train rmax = %d | dev rmax = %d\n', rmax, train_dataset.rmaxsize, dev_dataset.rmaxsize)

printf('num train = %d\n', train_dataset.size)
printf('num dev   = %d\n', dev_dataset.size)
printf('num test  = %d\n', test_dataset.size)

-- initialize model
local config = {
  input_dim = 300,
  mem_cols = 300,
  emb_vecs   = vecs,
--  structure  = 'NTM',
  read_heads = 1,
  task       = taskD,
  cont_dim = 250,
  structure = 'lstm',
}

--include('LSTMSimSemevalRank.lua')
--include('NTM.lstm.lua')
--include('NTM.lstm.simple.lua')
--include('NTM.lstm.entail.lua')
include('NTM.bilstm.entail.share.lua')
--include('NTM.bilstm.entail.share.stopword.lua')
--include('NTM.bilstm.entail.share.rawemb.lua')
--include('NTM.bilstm.entail.share.maxminmean.lua')
--include('NTM.NOlstm.entail.share.lua')
--include('NTM.bilstm.entail.share.extra.lua')
--include('NTM.lstm.entail.share.compareRwithH.lua')
--include('NTM.lstm.entail.share.convNgram.multipleread.lua')
local num_epochs = 60

local model = nil
local loadSave = false
if loadSave then
  include('./models/AlignMaxDropLeakyMulti.lua')
  include('./models/vgg_simpler.lua')
  --include('./models/AlignMaxDropLeakyMultiAll.lua')
  --include('./models/AlignMaxDropLeakyMultiAddUpDis.lua')
  --dofile('./models/vgg_simpler_tuneAlignDrop.lua')  
  model = torch.load("/scratch0/huah/savedModel/bestModelOnQA.ep1200.th")
  num_epochs = 1
else
  model = ntm.NTM(config)
end
--local model = ntm.LSTMSimSemevalRank(config)
-- number of epochs to train


-- print information
header('model configuration')
printf('max epochs = %d\n', num_epochs)
model:print_config()

if lfs.attributes(similarityMeasure.predictions_dir) == nil then
  lfs.mkdir(similarityMeasure.predictions_dir)
end

-- train
local train_start = sys.clock()
local best_dev_score = -1.0
local best_dev_model = model

-- threads
--torch.setnumthreads(4)
--print('<torch> number of threads in used: ' .. torch.getnumthreads())

header('Training model on data: ' .. taskD)

local id = 1532
print("Id: " .. id)

for i = 1, num_epochs do
  local start = sys.clock()
  print('--------------- EPOCH ' .. i .. '--- -------------')
  if not loadSave then
    model:trainCombineSeme(train_dataset)
    print('Finished epoch in ' .. ( sys.clock() - start) )
  end
  
  local dev_predictions = model:predict_dataset(dev_dataset)
  local dev_map_score = map(dev_predictions, dev_dataset.labelsReal, dev_dataset.boundary, dev_dataset.numrels)
  local dev_mrr_score = mrr(dev_predictions, dev_dataset.labelsReal, dev_dataset.boundary, dev_dataset.numrels)
  printf('-- dev map score: %.5f, mrr score: %.5f\n', dev_map_score, dev_mrr_score)

  --local dev_map_score = map(dev_predictions, dev_dataset.labels, dev_dataset.boundary, dev_dataset.numrels)
  --local dev_p30_score = p_30(dev_predictions, dev_dataset.labels, dev_dataset.boundary)
  --printf('-- dev map score: %.5f, p30 score: %.5f\n', dev_map_score, dev_p30_score)

  if not loadSave and dev_map_score >= best_dev_score then
    print("Saving best models onto Disk.")
    torch.save("/scratch0/huah/savedModel/bestModelOnWikiQA.ep" .. id ..".th", model)
    best_dev_score = dev_map_score
  end

    local predictions_save_path_dev = string.format(
        similarityMeasure.predictions_dir .. '/results-wikiQA-epoch-%d.%d.%s.dev.pred', i, id, taskD)
    local predictions_file_dev = torch.DiskFile(predictions_save_path_dev, 'w')
    print('Writing dev predictions to ' .. predictions_save_path_dev)
    for i = 1, dev_predictions:size(1) do
        local number = string.format('%.4f', dev_predictions[i])
	predictions_file_dev:writeString(number .. "\n")
    end
    predictions_file_dev:close()
    --local devNumber = os.execute("python ./predictions/eval.py ./predictions/dev.label " .. predictions_save_path_dev)
    --print(devNumber)
    --local test_map_score = map(test_predictions, test_dataset.labels, test_dataset.boundary, test_dataset.numrels)
    --local test_p30_score = p_30(test_predictions, test_dataset.labels, test_dataset.boundary)
    --printf('-- test map score: %.4f, p30 score: %.4f\n', test_map_score, test_p30_score)

    local test_predictions = model:predict_dataset(test_dataset)
    local test_map_score = map(test_predictions, test_dataset.labelsReal, test_dataset.boundary, test_dataset.numrels)
    local test_mrr_score = mrr(test_predictions, test_dataset.labelsReal, test_dataset.boundary, test_dataset.numrels)
    printf('-- test map score: %.5f, mrr score: %.5f\n', test_map_score, test_mrr_score)
    
    local predictions_save_path = string.format(	
	similarityMeasure.predictions_dir .. '/results-wikiQA-epoch-%d.%d.%s.test.pred', i, id, taskD)
    local predictions_file = torch.DiskFile(predictions_save_path, 'w')
    print('writing test predictions to ' .. predictions_save_path)
    for i = 1, test_predictions:size(1) do
      local number = string.format('%.4f', test_predictions[i])
      predictions_file:writeString(number .. "\n")
    end
    predictions_file:close()

    --local devNumber = os.execute("python ./predictions/eval.py ./predictions/test.label " .. predictions_save_path)
    --print(devNumber)
 -- end
end

print('finished training in ' .. (sys.clock() - train_start))
