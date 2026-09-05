import tensorflow as tf
import numpy as np

# Define the batch size
batch_size = 8

# Define the parameters
initializer = tf.keras.initializers.HeNormal(seed=1)
w1 = tf.Variable(initializer(shape=(2,3)))
b1 = tf.Variable(tf.zeros((3,)))
w2 = tf.Variable(initializer(shape=(3,1)))
b2 = tf.Variable(tf.zeros((1,)))

# Generate random data set by numpy
rng = np.random.RandomState(1)
data_size = 1000
x_train = rng.rand(data_size,2).astype(np.float32)
y_train = (
    x_train[:,0]+x_train[:,1] < 1
).astype(np.float32).reshape(-1,1)

# Define a new TensorFlow data set with x_train and y_train and split it into batches
dataset = tf.data.Dataset.from_tensor_slices(
    (x_train,y_train)
).batch(batch_size)

# Define loss function and optimizer
loss_fn = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001
)

# Define training loop, including forward propagation and back propagation
for epoch in range(5000):
    total_loss = 0.0
    counter = 0
    for x_batch, y_batch in dataset:
        with tf.GradientTape() as tape:
            # Define the forward propagation process
            a = tf.nn.relu(tf.matmul(x_batch,w1)+b1)
            y_pred = tf.sigmoid(tf.matmul(a,w2)+b2)
            
            # Compare the correct answer and the model's prediction
            loss = loss_fn(y_batch,y_pred)
        
        # Calculate gradients
        gradients = tape.gradient(
            loss,
            [w1,w2,b1,b2]
        )
        
        # Adjust parameters using gradients
        optimizer.apply_gradients(
            zip(gradients,[w1,w2,b1,b2])
        )
        
        total_loss += loss.numpy()
        counter +=1
    
    # Use average loss to check if loss decreases
    average_loss = total_loss/counter
    if epoch % 100 ==0:
        print("epoch:",epoch,"average loss:",average_loss)
