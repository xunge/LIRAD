This project refers to [https://github.com/albertan017/LLM4Decompile](https://github.com/albertan017/LLM4Decompile) and [https://github.com/AlongWY/sccdec](https://github.com/AlongWY/sccdec)

Our dataset is open source on huggingface [xunge/LIRAD](https://huggingface.co/datasets/xunge/LIRAD).

Our model is open source on huggingface [xunge/LIRAD-1.3B](https://huggingface.co/xunge/LIRAD-1.3B) and [xunge/LIRAD-6.7B](https://huggingface.co/xunge/LIRAD-6.7B).


### Evaluate

```bash
python run_evaluation_LIRAD_singleGPU.py \
--model_path xunge/LIRAD-1.3B \
--data_path ../decompile-eval/decompile-eval-executable-clang-obj.json
```

```bash
python run_evaluation_LIRAD_vllm.py \
--model_path xunge/LIRAD-1.3B \
--tokenizer_path xunge/LIRAD-1.3B \
--testset_path ../decompile-eval/decompile-eval-executable-clang-obj.json \
--output_path ../result/LIRAD-1.3B.json \
--output_result_path ../result/LIRAD-1.3B-result.json
```

