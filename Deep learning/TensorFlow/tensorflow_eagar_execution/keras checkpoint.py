library(keras)
use_implementation("tensorflow")

library(tensorflow)
tfe_enable_eager_execution(device_policy="silent")

library(tfdatasets)

# Prepare training and test sets ------------------------------------------

x_train < -
iris[1:120, c("Petal.Length", "Sepal.Length", "Petal.Width")] % > % as.matrix()
x_train < - k_constant(x_train)
y_train < -
iris[1:120, c("Sepal.Width")] % > % as.matrix()
y_train < - k_constant(y_train)

x_test < -
iris[121:150, c("Petal.Length", "Sepal.Length", "Petal.Width")] % > % as.matrix()
x_test < - k_constant(x_test)
y_test < -
iris[121:150, c("Sepal.Width")] % > % as.matrix()
y_test < - k_constant(y_test)

# Create datasets for training and testing --------------------------------

train_dataset < - tensor_slices_dataset(list(x_train, y_train)) % > %
dataset_batch(10)
test_dataset < - tensor_slices_dataset(list(x_test, y_test)) % > %
dataset_batch(10)

# Create model ------------------------------------------------------------

iris_regression_model < - function(name=NULL)
{
    keras_model_custom(name=name, function(self)
{
    self$dense1 < - layer_dense(units=32, input_shape=3)
self$dropout < - layer_dropout(rate=0.5)
self$dense2 < - layer_dense(units=1)

function(x, mask=NULL)
{
    self$dense1(x) % > %
self$dropout() % > %
self$dense2()
}
})
}

model < - iris_regression_model()

# Define loss function and optimizer --------------------------------------

mse_loss < - function(y_true, y_pred, x)
{
    mse < - tf$losses$mean_squared_error(y_true, y_pred)
mse
}

optimizer < - tf$train$AdamOptimizer()

# Set up checkpointing ----------------------------------------------------

checkpoint_dir < - "./checkpoints"
checkpoint_prefix < - file.path(checkpoint_dir, "ckpt")
checkpoint < -
tf$train$Checkpoint(optimizer=optimizer,
                    model=model)

n_epochs < - 10

# change to TRUE if you want to restore weights
restore < - FALSE

if (!restore) {
for (i in seq_len(n_epochs)) {
iter < - make_iterator_one_shot(train_dataset)
total_loss < - 0

until_out_of_range({
batch < -  iterator_get_next(iter)
x < - batch[[1]]
y < - batch[[2]]

with(tf$GradientTape() % as % tape, {
preds < - model(x)
loss < - mse_loss(y, preds, x)
})

total_loss < - total_loss + loss
gradients < - tape$gradient(loss, model$variables)

optimizer$apply_gradients(purrr::
transpose(list(gradients, model$variables)),
global_step = tf$train$get_or_create_global_step())

})

cat("Total loss (epoch): ", i, ": ", as.numeric(total_loss), "\n")

checkpoint$save(file_prefix=checkpoint_prefix)
}
} else {
    checkpoint$restore(tf$train$latest_checkpoint(checkpoint_dir))
}


# Get model predictions on test set ---------------------------------------

model(x_test)

iter < - make_iterator_one_shot(test_dataset)
until_out_of_range({
    batch < -  iterator_get_next(iter)
preds < - model(batch[[1]])
print(preds)
})
