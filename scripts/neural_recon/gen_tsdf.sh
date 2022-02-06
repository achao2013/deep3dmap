#Change data_path accordingly
python tools/data_gen/scannet.py --data_path /media/achao/Innov8/database/scannet-download/ --save_name all_tsdf_9 --window_size 9
python tools/data_gen/scannet.py --test --data_path /media/achao/Innov8/database/scannet-download/ --save_name all_tsdf_9 --window_size 9
