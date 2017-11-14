# KoreanLM-demo

## Prerequisites
### Required Torch libraries:
torch, nn, cutorch, cunn, lua-utf8 

### Trained model:


## How to execute:
`$th demo_main_sylgen.lua`

It basically requires CUDA   

## Demo Examples:
    using CUDA on GPU 0...
	--- Loading vocab : exp/vocab.t7
	Word vocab size: 2189
	-- Building network --
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
	유럽의
	LSTM : resetStates
	LSTM : resetStates
	
	
	유럽의 경우 이 지역에서 이 지역에서 가장 많은 수출이 가능하다<eos>
