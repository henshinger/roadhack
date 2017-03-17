from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json

# create the base pre-trained model
# base_model = ResNet50(weights='imagenet', include_top=False)

# # add a global spatial average pooling layer
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# # let's add a fully-connected layer
# x = Dense(1024, activation='relu')(x)
# # and a logistic layer -- let's say we have 200 classes
# predictions = Dense(1, activation='sigmoid')(x)

# # this is the model we will train
# model = Model(input=base_model.input, output=predictions)

# # first: train only the top layers (which were randomly initialized)
# # i.e. freeze all convolutional InceptionV3 layers
# for layer in base_model.layers:
#     layer.trainable = False

# # compile the model (should be done *after* setting layers to non-trainable)
# model.compile(optimizer='rmsprop', loss='binary_crossentropy')

json_file = open('resnet50.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("resnet50.hdf5")
print("Loaded model from disk")

loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# train the model on the new data for a few epochs
train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    "train",
    target_size=(224, 224),
    batch_size=100,
    class_mode='binary')


print(loaded_model.evaluate_generator(train_generator,100))
print(loaded_model.evaluate_generator(train_generator,100))
print(loaded_model.evaluate_generator(train_generator,100))
print(loaded_model.evaluate_generator(train_generator,100))
print(loaded_model.evaluate_generator(train_generator,100))
print(loaded_model.evaluate_generator(train_generator,100))
print(loaded_model.evaluate_generator(train_generator,100))
print(loaded_model.evaluate_generator(train_generator,100))
print(loaded_model.evaluate_generator(train_generator,100))
print(loaded_model.evaluate_generator(train_generator,100))
print(loaded_model.evaluate_generator(train_generator,100))

# print(loaded_model.metrics_names)

# [0.27350956201553345, 0.9100000262260437]
# [0.29944175481796265, 0.81999999284744263]
# [0.2229771614074707, 0.93000000715255737]
# [0.25782492756843567, 0.93000000715255737]
# [0.31591460108757019, 0.88999998569488525]
# [0.31199362874031067, 0.9100000262260437]
# [0.23959709703922272, 0.92000001668930054]
# [0.23185257613658905, 0.9100000262260437]
# [0.28688430786132812, 0.88999998569488525]
# [0.28288025946556766, 0.87974682261672199]
# [0.25617155432701111, 0.89999997615814209]


#[0.26517000794410706, 0.92000001668930054]

#[0.24338169395923615, 0.93000000715255737]


# [1.9927982091903687, 0.875]
# ['loss', 'acc']

# [1.1159669160842896, 0.93000000715255737]
# ['loss', 'acc']

# [2.3913576602935791, 0.85000002384185791]
# ['loss', 'acc']

# [0.63769549131393433, 0.95999997854232788]
# ['loss', 'acc']

# [1.5942384004592896, 0.89999997615814209]
# ['loss', 'acc']


# model.save_weights('resnet50.hdf5')
# with open('resnet50.json', 'w') as f:
#     f.write(model.to_json())

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
# for i, layer in enumerate(base_model.layers):
#    print(i, layer.name)

# # we chose to train the top 2 inception blocks, i.e. we will freeze
# # the first 172 layers and unfreeze the rest:
# for layer in model.layers[:-3]:
#    layer.trainable = False
# for layer in model.layers[-3:]:
#    layer.trainable = True

# # we need to recompile the model for these modifications to take effect
# # we use SGD with a low learning rate
# from keras.optimizers import SGD
# model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy')

# # we train our model again (this time fine-tuning the top 2 inception blocks
# # alongside the top Dense layers
# model.fit_generator(...)
