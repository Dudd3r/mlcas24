import data_downloader
import data_formatter
import resnet_mono

if __name__ == "__main__":
    
    print("[INFO] Executing autorun.")
    data_downloader.run()
    data_formatter.run()
    resnet_mono.run("resnet5_200_fin")