from landmark import cnn_classifier
import numpy as np
from landmark import utils

land_mark='10th Rib Midspine'

example_obj=utils.read_obj('/gpfs/projects/datascience/shared/OtherScans/3DMDScan.obj')

xs=np.array([example_obj])

model=cnn_classifier.build_model(xs.shape[1],-999)
model.compile(loss='mse',optimizer='adam')
model.summary()

model.load_weights('tests.h5')

prediction=model.predict(xs)
lm={land_mark:prediction[0,:]}
utils.write_landmark_xml('test.xml',lm)
