import os 
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import tensorflow as tf 
import numpy as np
from network import MidiNet

tf.logging.set_verbosity(tf.logging.ERROR)
tf.reset_default_graph()

# train_path = 'mario_dataset.npy'
# train_path = 'piano_dataset_big.npy'
# train_path = '24bsb.npy'
train_path = 'piano_data_24.npy'
experimentNUMBER = 25
model_path = 'experiments/EXPERIMENT{}/model/'.format(experimentNUMBER)

LOAD = False

EPOCHS = 100
BATCH_SIZE = 72

#! TRAINING PARAMETERS
NG = 2
ND = 5
LAMBDA1 = 0.1
LAMBDA2 = 1.0
LAMBDAGP = 10
LR = 1e-4

if __name__ == '__main__':
    dataset = np.load(train_path)
    model = MidiNet(100, [16, 128], method='vanilla')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if LOAD:
            try:
                model.restore(sess, model_path+'model')
                print("\nModel loaded...\n")
            except Exception as e:
                print(e)
                exit()


        for epoch in range(EPOCHS):
            for episode in range(len(dataset)//BATCH_SIZE):
                data = dataset[episode*BATCH_SIZE:min((episode+1)*BATCH_SIZE, len(dataset))]
                real_bars = data[:, 0]
                prev_bars = data[:, 1]
                train_noise = np.random.normal(loc=0.0, scale=1.0, size=(BATCH_SIZE, 100))
                d_loss, g_loss = model.train(sess, noise=train_noise, data=real_bars, prev_data=prev_bars,
                                    n_g=NG, n_d=ND, lambda1=LAMBDA1, lambda2=LAMBDA2, lambda_gp=LAMBDAGP, lr=LR)
                print("Epoch:%3.d\tEpisode:%6.d\tD_loss:%.7f\tG_loss:%.7f" % (epoch+1, episode+1, d_loss, g_loss))
            model.save(sess, model_path+'model')
