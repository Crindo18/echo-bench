"""echo_core: the code that ships on the ECHO device (blueprint section 2.2).

It will hold the audio frontend, the ONNX Runtime session, the prototype
classifier, the H-KWS interface and enrollment. It must never import torch,
espnet, librosa or pandas, because the Raspberry Pi never installs them;
tests/test_architecture.py enforces this.

M0 only creates the package. Modules arrive with their milestones:
M1 model_card.py and runtime/session.py, M4 frontend/, M5 classify/,
M7 kws/, M8 enroll/, routing.py and tts/.
"""

__version__ = "0.1.0"
