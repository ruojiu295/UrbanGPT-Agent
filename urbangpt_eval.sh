# to fill in the following path to evaluation!
output_model=./checkpoints/UrbanGPT_pretrained
datapath=./ST_data_urbangpt/NYC_taxi_cross-region/NYCtaxi_zeroshot.json
st_data_path=./ST_data_urbangpt/NYC_taxi_cross-region/NYCtaxi_zeroshot_pkl.pkl
res_path=./result_test/cross-region/NYC_taxi
start_id=0
end_id=51920
num_gpus=8

python ./urbangpt/eval/run_urbangpt.py --model-name ${output_model}  --prompting_file ${datapath} --st_data_path ${st_data_path} --output_res_path ${res_path} --start_id ${start_id} --end_id ${end_id} --num_gpus ${num_gpus}