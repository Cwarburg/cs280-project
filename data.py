import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory="SoccerNet")
mySoccerNetDownloader.password = "s0cc3rn3t"
mySoccerNetDownloader.downloadRAWVideo(dataset="SoccerNet") # download 720p Videos
