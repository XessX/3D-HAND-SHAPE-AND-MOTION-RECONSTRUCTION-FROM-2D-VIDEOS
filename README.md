The project is My Final Project issued from McMaster Universities Computing and Software Department. Dr. YingYing Wang is being the supervisor of my project and the project was handled fully by me.<br />
The project works using ACR https://semanticdh.github.io/ACR/ and Mano https://mano.is.tue.mpg.de/ and the videos were taken from this pre-trained Sign language dataset https://how2sign.github.io/ <br />
Instructions to install and work- <br />
(works over python 3.8.8) <br />
`pip install python==3.8.8"
 pip install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html
 pip install -r requirements.txt
 pip install mediapipe
`
then run
`python main.py`

This provides the result by taking the input of a video and providing an output of the 3d hand.
