
from save_query_results import run_query_results

if __name__ == "__main__":
	PATH_DRIVE = "C:/Users/OPER/OneDrive - Israel Aerospace Industries"

    # input radiu featurs
	gallery_dirs = [
		"D:/OrthoIsr/data/crop1000",
		"D:/OrthoIsr/data/crop1500",
		"D:/OrthoIsr/data/crop2000",
	]
	# result
	crop_folders = [
		"D:/OrthoIsr/data/crop1000",
		"D:/OrthoIsr/data/crop1500",
		"D:/OrthoIsr/data/crop2000",
	]

	for gallery_dir, crop_folder in zip(gallery_dirs, crop_folders):
		run_query_results(
			query_dir=f"{PATH_DRIVE}/Frames_ofek/session_2026-01-04_14-47-20_309/frames",
			session_name="309",
			gallery_dir=gallery_dir,
			crop_folder=crop_folder,
			repo="C:/github/RADIO",
			model_version="c-radio_v4-h",
			query_step=90,
			query_start=1600,
			query_end=13000,
		)
