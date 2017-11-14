# KoreanLM-demo

## Prerequisites
### Required Torch libraries:
torch, nn, cutorch, cunn, lua-utf8 

### Trained model:
Download from ...
exp/lm-rnn2048x2-v2-wd0-dw1/lm_char_epoch43.00.t7

## How to execute:
`$th demo_main_sylgen.lua`

It basically requires CUDA   

## Demo Examples:
    using CUDA on GPU 0...
	--- Loading vocab : exp/vocab.t7
	Word vocab size: 2189
	-- Building network : exp/lm-rnn2048x2-v2-wd0-dw1/lm_char_epoch43.00.t7
	======================================
	Please enter a front part of sentence :
	더
	LSTM : resetStates
	LSTM : resetStates
	
	
	더 많은 것을 보고 있다<eos>
	
	======================================
	Please enter a front part of sentence :
	위
	LSTM : resetStates
	LSTM : resetStates
	
	
	위원회는 이 사업을 추진하고 있다<eos>
	
	======================================
	Please enter a front part of sentence :
	이날
	LSTM : resetStates
	LSTM : resetStates


	이날 오전 10시 30분께 이 사건을 수사하고 있다<eos>

	======================================
	Please enter a front part of sentence :
	비가
	LSTM : resetStates
	LSTM : resetStates


	비가 내리는 것은 이번이 처음이다<eos>
	======================================
	Please enter a front part of sentence :
	유럽의
	LSTM : resetStates
	LSTM : resetStates
	
	
	유럽의 경우 이 지역에서 이 지역에서 가장 많은 수출이 가능하다<eos>
