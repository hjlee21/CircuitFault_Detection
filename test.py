import os
import tensorflow as tf
import math
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
x_train = tf.random.normal(shape = (100, 1), dtype = tf.float32)
y_train = tf.math.sin(2 * math.pi * x_train) / (2 * math.pi) + 1

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=32, activation='relu', input_shape=(1,)),
    # tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=1000, verbose=1)

"""
EPOCHS = 60
LR = 0.001

model = Linearpredictor()
optimizer = tf.keras.optimizers.SGD(learning_rate = LR)
loss = tf.keras.losses.MeanSquaredError()

for epoch in range(EPOCHS):
    total_loss=0.0
    for x, y in zip(x_train, y_train):
        x = tf.reshape(x, (1, 1))
        y = tf.reshape(y, (1, 1))
        #tf.Tensor([[-0.6477713]], shape=(1, 1), dtype=float32)
        #y -> tf.Tensor([0.8416938], shape=(1,), dtype=float32
        with tf.GradientTape() as tape:
            y_pred = model(x)
            #y_pred -> tf.Tensor([[1.0077525]], shape=(1, 1), dtype=float32)

            loss_value = loss(y, y_pred)
            #loss_value -> tf.Tensor(0.018896561, shape=(), dtype=float32)

        gradients = tape.gradient(loss_value, model.trainable_variables) 
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        total_loss += loss_value.numpy()

    average_loss = total_loss / len(x_train)  # 평균 손실값 계산
    print('Epoch: ', epoch + 1, 'Loss: ', average_loss)

"""
# print('Epoch: ', epoch + 1, 'Loss: ', loss_value.numpy())
x_test = tf.random.normal(shape = (100, 1), dtype = tf.float32)
y_test = model(x_test)




x_points = tf.linspace(-2, 2, 100)
y_points = tf.math.sin(2 * math.pi * x_points) / (2 * math.pi) + 1
plt.plot(x_points, y_points, 'r')

plt.scatter(x_test, y_test)
plt.show()