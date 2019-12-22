import tensorflow as tf 
import numpy as np 


class MidiNet:

    def __init__(self, session, noise_shape, output_shape, method='vanilla', device='/gpu:0', summary_dir='./logs'):
        self.session = session
        self.device = device
        self.method = method.lower()
        assert self.method=='vanilla' or self.method=='wgan', "Method chosen for the model should be either 'Vanilla' or 'WGAN'."
        self.weight_init = tf.initializers.random_normal(mean=0.0, stddev=0.02)
        self.dim_x, self.dim_y = output_shape

        #* Placeholders
        
        self.Z = tf.placeholder(tf.float32, shape=[None, noise_shape], name='noise')
        self.X = tf.placeholder(tf.float32, shape=[None, self.dim_x, self.dim_y, 1], name='real_data')
        self.X_prev = tf.placeholder(tf.float32, shape=[None, self.dim_x, self.dim_y, 1], name='prev_data')
        self.lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.training = tf.placeholder(tf.bool, name='batchnorm_condition')
        self.lambda1 = tf.placeholder(tf.float32, shape=[], name='lambda1')
        self.lambda2 = tf.placeholder(tf.float32, shape=[], name='lambda2')
        self.lambda_gp = tf.placeholder(tf.float32, shape=[], name='lambda_gp')


        ##* Feeding procedure
        #? Get conditioner layer values first
        self.conditioner(self.X_prev, filters=256)
        self.fake = self.generator(self.Z, filters=256)
        fake_logit, fake_result, disc1_fake = self.discriminator(self.fake, reuse=False)
        real_logit, real_result, disc1_real = self.discriminator(self.X)

        self.Z_sum = tf.summary.histogram("Z", self.Z)
        self.real_logit_sum = tf.summary.histogram("real_logit", real_logit)
        self.fake_logit_sum = tf.summary.histogram("fake_logit", fake_logit)
        self.fake_sum = tf.summary.image("fake", self.fake)

        self.g_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")
        self.d_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")
        self.c_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="conditioner")
        for c in self.c_variables:
            self.g_variables.append(c)

        #* Feature Matching
        repr_loss = tf.reduce_mean(self.fake, axis=0) - tf.reduce_mean(self.X, axis=0)
        repr_loss = tf.multiply(tf.nn.l2_loss(repr_loss), self.lambda1)

        layer_loss = tf.reduce_mean(disc1_fake, axis=0) - tf.reduce_mean(disc1_real, axis=0)
        layer_loss = tf.multiply(tf.nn.l2_loss(layer_loss), self.lambda2)

        #* GAN Optimization Method: Vanilla / Wasserstein
	    ##? Vanilla GAN 
        if self.method == 'vanilla':
            # # ! Classical GAN Losses
            # eps = 1e-08
            # self.d_loss = -tf.reduce_mean(tf.log(real_result + eps) + tf.log(tf.constant(1.0)- fake_result + eps), name='discriminator_loss')
            # self.g_loss = -(tf.reduce_mean(tf.log(fake_result + eps) + repr_loss + layer_loss)) 
            self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logit, labels=0.9*tf.ones_like(real_result)))
            self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit, labels=tf.zeros_like(fake_result)))
            self.g_loss0 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit, labels=tf.ones_like(fake_result))) 
            self.d_loss = self.d_loss_real + self.d_loss_fake
            self.g_loss = self.g_loss0 + repr_loss + layer_loss

            self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
            self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        #* Optimize variables
            self.d_opt = tf.train.AdamOptimizer(self.lr).minimize(loss=self.d_loss, var_list=self.d_variables)
            self.g_opt = tf.train.AdamOptimizer(self.lr).minimize(loss=self.g_loss, var_list=self.g_variables)
        
        #? Wasserstein GAN
        elif self.method == 'wgan':

            self.d_loss = tf.reduce_mean(real_logit - fake_logit)
            self.g_loss = -tf.reduce_mean(fake_logit) + repr_loss + layer_loss
            
            self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
            self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.d_opt = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(loss=self.d_loss, var_list=self.d_variables)
                self.g_opt = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(loss=self.g_loss, var_list=self.g_variables)
            #* WGAN Clipping operation
            C_MAX = tf.constant(0.01, dtype=tf.float32, name='Clipping_max_factor')
            C_MIN = tf.constant(-0.01, dtype=tf.float32, name='Clipping_min_factor')
            self.clipping_op = [var.assign(tf.clip_by_value(var, C_MIN, C_MAX)) for
                                            var in self.d_variables]

        elif self.method == 'wgangp':
                        
            #* Gradient Penalty
            epsilon = tf.random.normal(shape=[1], mean=0.0, stddev=1)
            self.X_tilda = tf.multiply(epsilon,self.X) + tf.multiply((1-epsilon), self.fake)
            with tf.GradientTape() as t:
                t.watch(self.X_tilda)
                tilda_logit, _, _ = self.discriminator(self.X_tilda)
            self.gradients = tf.gradients(tilda_logit, self.X_tilda, name='grads_tilda')[0]
            self.gradient_penalty = tf.multiply(tf.nn.l2_loss(self.gradients) - 1, self.lambda_gp)

            self.d_loss = tf.reduce_mean(real_logit - fake_logit) + self.gradient_penalty
            self.g_loss = -tf.reduce_mean(fake_logit) + repr_loss + layer_loss
            
            self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
            self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.d_opt = tf.train.AdamOptimizer(self.lr, beta1=0.1).minimize(loss=self.d_loss, var_list=self.d_variables)
                self.g_opt = tf.train.AdamOptimizer(self.lr, beta1=0.1).minimize(loss=self.g_loss, var_list=self.g_variables)


        self.d_summary = tf.summary.merge([self.Z_sum, self.real_logit_sum, self.d_loss_sum])
        self.g_summary = tf.summary.merge([self.Z_sum, self.fake_logit_sum, self.g_loss_sum, self.fake_sum])
        self.writer = tf.summary.FileWriter(summary_dir, self.session.graph)
        self.saver = tf.train.Saver()
        self.counter = 0

    def conditioner(self, X, filters=16):
        with tf.device(self.device):
            with tf.variable_scope('conditioner', reuse=tf.AUTO_REUSE):
                self.conditioner1 = tf.layers.conv2d(X, filters=filters, kernel_size=(1,self.dim_y), strides=(1,2), activation=None, kernel_initializer=self.weight_init, name='conditioner_conv1')
                self.conditioner1 = tf.nn.leaky_relu(tf.layers.batch_normalization(self.conditioner1, scale=False, training=self.training, name='conditioner_bn1'))

                self.conditioner2 = tf.layers.conv2d(self.conditioner1, filters=filters, kernel_size=(2,1), strides=(2,2), activation=None, kernel_initializer=self.weight_init, name='conditioner_conv2')
                self.conditioner2 = tf.nn.leaky_relu(tf.layers.batch_normalization(self.conditioner2, scale=False, training=self.training, name='conditioner_bn2'))

                self.conditioner3 = tf.layers.conv2d(self.conditioner2, filters=filters, kernel_size=(2,1), strides=(2,2), activation=None, kernel_initializer=self.weight_init, name='conditioner_conv3')
                self.conditioner3 = tf.nn.leaky_relu(tf.layers.batch_normalization(self.conditioner3, scale=False, training=self.training, name='conditioner_bn3'))

                self.conditioner4 = tf.layers.conv2d(self.conditioner3, filters=filters, kernel_size=(2,1), strides=(2,2), activation=None, kernel_initializer=self.weight_init, name='conditioner_conv4')
                self.conditioner4 = tf.nn.leaky_relu(tf.layers.batch_normalization(self.conditioner4, scale=False, training=self.training, name='conditioner_bn4'))


    def generator(self, Z, filters=16):
        with tf.device(self.device):
            with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
                generatorFC1 = tf.layers.dense(Z, 1024, activation=None, name='generator_fc1')
                generatorFC1 = tf.nn.relu(tf.layers.batch_normalization(generatorFC1, scale=False, training=self.training, name='generator_bn1'))

                generatorFC2 = tf.layers.dense(generatorFC1, 256, activation=None, name='generator_fc2')
                generatorFC2 = tf.nn.relu(tf.layers.batch_normalization(generatorFC2, scale=False, training=self.training, name='generator_bn2'))

                generatorFC2 = tf.reshape(generatorFC2, [-1, 2, 1, 128])
                
                #? Conditioner4 + reshaped FC2 Output
                # deconv_input1 = tf.concat([generatorFC2, self.conditioner4], 3)
                deconv_input1 = generatorFC2
                self.generator1 = tf.layers.conv2d_transpose(deconv_input1, filters=filters, kernel_size=(2,1), strides=(2,1), activation=None, kernel_initializer=self.weight_init, name='generator_deconv1')
                self.generator1 = tf.nn.relu(tf.layers.batch_normalization(self.generator1, scale=False, training=self.training, name='generator_bn3'))

                deconv_input2 = tf.concat([self.generator1, self.conditioner3], 3)
                deconv_input2 = self.generator1
                self.generator2 = tf.layers.conv2d_transpose(deconv_input2, filters=filters, kernel_size=(2,1), strides=(2,1), activation=None, kernel_initializer=self.weight_init, name='generator_deconv2')
                self.generator2 = tf.nn.relu(tf.layers.batch_normalization(self.generator2, scale=False, training=self.training, name='generator_bn4'))
                
                # deconv_input3 = tf.concat([self.generator2, self.conditioner2], 3)
                deconv_input3 = self.generator2
                self.generator3 = tf.layers.conv2d_transpose(deconv_input3, filters=filters, kernel_size=(2,1), strides=(2,1), activation=None, kernel_initializer=self.weight_init, name='generator_deconv3')
                self.generator3 = tf.nn.relu(tf.layers.batch_normalization(self.generator3, scale=False, training=self.training, name='generator_bn5'))
                
                deconv_input4 = tf.concat([self.generator3, self.conditioner1], 3)
                self.generator4 = tf.layers.conv2d_transpose(deconv_input4, filters=1, kernel_size=(1,self.dim_y), strides=(1,2), activation=tf.nn.sigmoid, kernel_initializer=self.weight_init, name='generator_deconv4')

        return self.generator4
        
    def discriminator(self, X, reuse=True):
        
        with tf.device(self.device):
            with tf.variable_scope('discriminator', reuse=reuse):
                self.discriminator1 = tf.layers.conv2d(X, filters=14, kernel_size=(4,89), strides=(1,1), activation=None, kernel_initializer=self.weight_init, name='discriminator_conv1') # kernel_size=(4, 89) for dim_y=128, kernel_size=(4, 45) for dim_y=84
                # self.discriminator1 = tf.nn.leaky_relu(tf.layers.batch_normalization(self.discriminator1, scale=False, training=True, name='discriminator_bn1'))
                self.discriminator1 = tf.nn.leaky_relu(self.discriminator1)

                self.discriminator2 = tf.layers.conv2d(self.discriminator1, filters=64, kernel_size=(4,1), strides=(1,1), activation=None, kernel_initializer=self.weight_init, name='discriminator_conv2')
                # self.discriminator2 = tf.nn.leaky_relu(tf.layers.batch_normalization(self.discriminator2, scale=False, training=True, name='discriminator_bn2'))
                self.discriminator2 = tf.nn.leaky_relu(self.discriminator2)

                # self.discriminator3 = tf.layers.conv2d(self.discriminator1, filters=77, kernel_size=(4,1), strides=(1,1), activation=None, kernel_initializer=self.weight_init, name='discriminator_conv3')
                # self.discriminator3 = tf.nn.leaky_relu(tf.layers.batch_normalization(self.discriminator3, scale=False, training=True, name='discriminator_bn3'))

                # self.discriminator3 = tf.layers.conv2d(self.discriminator1, filters=77, kernel_size=(4,1), strides=(1,1), activation=None, kernel_initializer=self.weight_init, name='discriminator_conv3')
                # self.discriminator3 = tf.nn.leaky_relu(tf.layers.batch_normalization(self.discriminator3, scale=False, training=True, name='discriminator_bn3'))
                # self.discriminator3 = tf.nn.leaky_relu(self.discriminator3)
                # print(self.discriminator3)
                self.discriminator2 = tf.layers.flatten(self.discriminator2, name='flatten')
                # self.discriminator3 = tf.reshape(self.discriminator3, [-1, 17920])
                self.discriminator3 = tf.layers.dense(self.discriminator2, 1024, activation=None, name='discriminator_fc1')
                # self.discriminator3 = tf.nn.leaky_relu(tf.layers.batch_normalization(self.discriminator3, scale=False, training=True, name='discriminator_bn3'))
                self.discriminator3 = tf.nn.leaky_relu(self.discriminator3)

                # self.discriminator3 = tf.reshape(self.discriminator3, [-1, 17920])
                # self.discriminator3 = tf.layers.flatten(self.discriminator3, name='flatten')
                logit = tf.layers.dense(self.discriminator3, 1, activation=None, name='discriminator_fc2')
                output = tf.nn.sigmoid(logit)
        
        return logit, output, self.discriminator1

    def train(self, noise, data, prev_data, n_g=2, n_d=1, lambda1=0.01, lambda2=0.1, lambda_gp=10, lr=1e-5):
        d_loss = 0
        g_loss = 0

        if self.method == 'vanilla':
            for _ in range(n_d):
                d_loss_, _, summary = self.session.run([self.d_loss, self.d_opt, self.d_summary], feed_dict={self.Z: noise, self.X: data, self.X_prev: prev_data,
                                                                            self.lr: lr, self.training: True,
                                                                            self.lambda1: lambda1, self.lambda2: lambda2})
                d_loss += d_loss_
                self.writer.add_summary(summary, self.counter)

            for _ in range(n_g):
                g_loss_, _, summary = self.session.run([self.g_loss, self.g_opt, self.g_summary], feed_dict={self.Z:noise, self.X: data, self.X_prev: prev_data, 
                                                self.lr: lr, self.training: True,
                                                self.lambda1: lambda1, self.lambda2: lambda2})
                g_loss += g_loss_
                self.writer.add_summary(summary, self.counter)

        elif self.method == 'wgan':
            for _ in range(n_d):
                d_loss, _, _, summary = self.session.run([self.d_loss, self.d_opt, self.clipping_op, self.d_summary], feed_dict={self.Z: noise, self.X: data, self.X_prev: prev_data, #, self.clipping_op
                                                                                              self.lr: lr, self.training: True,
                                                                                              self.lambda1: lambda1, self.lambda2: lambda2, self.lambda_gp: lambda_gp})
                d_loss += d_loss_
                self.writer.add_summary(summary, self.counter)
            for _ in range(n_g):
                g_loss, _, _, summary = self.session.run([self.g_loss, self.g_opt, self.clipping_op, self.g_summary], feed_dict={self.Z:noise, self.X: data, self.X_prev: prev_data, 
                                                self.lr: lr, self.training: True,
                                                self.lambda1: lambda1, self.lambda2: lambda2, self.lambda_gp: lambda_gp})
                g_loss += g_loss_
                self.writer.add_summary(summary, self.counter)
        elif self.method == 'wgangp':
            for _ in range(n_d):
                d_loss, _, summary = self.session.run([self.d_loss, self.d_opt, self.d_summary], feed_dict={self.Z: noise, self.X: data, self.X_prev: prev_data, #, self.clipping_op
                                                                                              self.lr: lr, self.training: True,
                                                                                              self.lambda1: lambda1, self.lambda2: lambda2, self.lambda_gp: lambda_gp})
                d_loss += d_loss_
                self.writer.add_summary(summary, self.counter)
            for _ in range(n_g):
                g_loss, _, summary = self.session.run([self.g_loss, self.g_opt, self.g_summary], feed_dict={self.Z:noise, self.X: data, self.X_prev: prev_data, 
                                                self.lr: lr, self.training: True,
                                                self.lambda1: lambda1, self.lambda2: lambda2, self.lambda_gp: lambda_gp})
                g_loss += g_loss_
                self.writer.add_summary(summary, self.counter)
        self.counter += 1
        g_loss /= n_g
        d_loss /= n_d

        return d_loss, g_loss

    def generate(self, noise, prev_sample):
        return self.session.run(self.fake, feed_dict={self.Z: noise, self.X_prev: prev_sample, self.training: False})

    def save(self, filepath):
        return self.saver.save(self.session, filepath)
    
    def restore(self, filepath):
        return self.saver.restore(self.session, filepath)