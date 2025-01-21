import os
import gdown
import traceback
import zipfile

def download_csv(link_zipped_csv, gdrive_link, zipped_file):

    file_id = link_zipped_csv.split('/')[-2]  # Take the file_id (Ex. "https://drive.google.com/file/d/1BMj4BGXxIMzsd-GYSAEMpB7CF0XB87UT/view?usp=sharing" => file_id: 1BMj4BGXxIMzsd-GYSAEMpB7CF0XB87UT)
    download_link = gdrive_link + file_id # Create the path for gdown (Ex. https://drive.google.com/uc?id={YOUR_FILE_ID})

    try:
        if not os.path.exists(zipped_file):

            gdown.download(
                download_link,
                zipped_file,
                quiet=False
                )
        else:
            print("CSV file already downloaded!")


    except Exception as error:
        print("An error occured:", error)
        traceback.print_exc()

def unzip_csv(csv_zip, csv_dir):

    try:
        if not os.path.exists(csv_dir):
            os.mkdir(csv_dir) # Create the csv_dir directory where we extract files (if not exists)

        if len(os.listdir(csv_dir)) == 0:
            with zipfile.ZipFile(csv_zip, 'r') as zip:
                filelist = zip.namelist() # list of the file inside zip : ['csv /multilingual_nli_test_df.csv', 'csv /tweet_emotions.csv', ...]

                # Iterate over all file in the zip file to extract them
                for filename in filelist:
                    zip.extract(filename, csv_dir) # Extract the file inside the csv_dir

        else:
            print("CSV file already unzipped!")


    except Exception as error:
        print("An error occured:", error)
        traceback.print_exc()