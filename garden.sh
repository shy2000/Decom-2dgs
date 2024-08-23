CUDA_VISIBLE_DEVICES=0 python train.py -s ~/data/360_v2/garden -m output/m360/garden
python train.py -s ~/data/scan6/ -m output/scan6 --start_checkpoint output/scan6/chkpnt30000.pth  --include_feature --read_instance
python train.py -s ~/data/scan6/ -m output/scan6 --start_checkpoint /home/shenhongyu/2d-gaussian-recon/output/chkpnt/replica/scan6/chkpnt30000.pth  --include_feature --read_instance
python train.py -s ~/data/scan6/ -m output/scan6 --start_checkpoint /home/shenhongyu/2d-gaussian-recon/output/chkpnt/replica/scan6/chkpnt30000.pth  --include_feature --read_instance --contrastive
python render_ids.py -s ~/data/scan6/ -m output/scan6 --start_checkpoint output/chkpnt/replica/scan6/chkpnt_contrastive_30000.pth