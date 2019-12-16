# uml-dl-project
Deep learning (COMP.5300) course project.

Project report: https://github.com/z00bean/uml-dl-project/blob/master/Report-Hand_Sign_Recognition.pdf

Slides: https://github.com/z00bean/uml-dl-project/blob/master/Presentation-FINAL-Zubin.pdf

opencv-python must be installed.

To run: 

    $python3 main.py

Some parameters which can be configured are:

    --scorethreshold (default is 0.5)
    --width (default is 320)
    --height (default is 180)
    
Datasets are not in this repository. So, they have to be downloaded before retraining.
(download_egoHands_dataset_clean.py script downloads and preprocesses the Egohand dataset. The signlanguage dataset has to be downloaded from here.)

MobileNetSSDv1 (Tensorflow model zoo) is used for detecting the hands from images, and a simple 2 layer CNN with max-pooling is used to detect the signs. Links of libraries and tutorials are given in the report.

