import os
import random
import time

from util import *
from parameters import *
from tensorflow.keras import datasets, layers, models

class CNN:
    def __init__(self):
        self.info_list = []
        self.file_dict = {}
        self.input_data = []
        self.label_data = []
        self.train_input = []
        self.train_label = []
        self.validation_input = []
        self.validation_label = []
        self.test_input = []
        self.test_label = []
        self.region = []
        self.name = []
        self.model = ""

    def load_data(self, path_info, path_map):
        with open(path_info, "rb") as file:
            self.info_list = pickle.load(file)
        self.input_data, self.label_data, self.region, self.name = load_data_CNN(map_dict, self.info_list, path_map)

    def shuffle_data(self, distribution=[6, 2, 3], total=10, plot=False):
        # shuffle process
        lst = [str(i) for i in range(total + 1)]
        random.shuffle(lst)
        train_region = []
        validation_region = []
        test_region = []
        for i in range(distribution[0] + distribution[1]):
            train_region.append(lst.pop())
        for i in range(distribution[2]):
            test_region.append(lst.pop())

        # set train/validation/test index
        train_index = []
        validation_index = []
        test_index = []
        for i in range(len(self.region)):
            re = self.region[i]
            if (re[re.index("c")+1:re.index("e")] in test_region) and (re[re.index("m")+1:] in test_region):
                test_index.append(i)
            else:
                train_index.append(i)
        random.shuffle(train_index)
        for i in range((distribution[1]*len(train_index))//(distribution[0]+distribution[1])):
            validation_index.append(train_index.pop())

        # assign data
        self.train_input = self.input_data[train_index]
        self.train_input = NCWH_to_NWHC(self.train_input)
        self.validation_input = self.input_data[validation_index]
        self.validation_input = NCWH_to_NWHC(self.validation_input)
        self.test_input = self.input_data[test_index]
        self.test_input = NCWH_to_NWHC(self.test_input)

        self.train_label = self.label_data[train_index]
        self.train_label = self.train_label.reshape((self.train_label.shape[0], 1))
        self.validation_label = self.label_data[validation_index]
        self.validation_label = self.validation_label.reshape((self.validation_label.shape[0], 1))
        self.test_label = self.label_data[test_index]
        self.test_label = self.test_label.reshape((self.test_label.shape[0], 1))

        if plot:
            x = np.arange(0, self.train_input.shape[2], 1)
            y = np.arange(0, self.train_input.shape[2], 1)
            X, Y = np.meshgrid(x, y)

            print("plot examples of training data:")
            time.sleep(0.5)
            for n in tqdm(range(self.train_input.shape[0])):
                fig = figure.Figure(figsize=(15, 7.5))
                ax = fig.subplots(3, 6)
                for i in range(3):
                    for j in range(6):
                        _map = self.train_input[n, :, :, 6*i+j]
                        ax[i][j].pcolormesh(X, Y, _map, vmin=np.min(_map), vmax=np.max(_map), shading='auto')
                        if i == 0 and j == 0:
                            ax[i][j].set_ylabel("unit")
                        if i == 1 and j == 0:
                            ax[i][j].set_ylabel("sn")
                        if i == 2 and j == 0:
                            ax[i][j].set_ylabel("rsn")
                        if i == 0 and j == 1:
                            ax[i][j].set_title(r"nrn: %s" % (self.name[train_index[n]][0]))
                        if i == 0 and j == 4:
                            ax[i][j].set_title(r"nrn: %s" % (self.name[train_index[n]][1]))

                fig.suptitle("train_input_example: " + str(self.train_label[n][0]==1))
                fig.tight_layout(rect=(0, 0, 1, 0.99))
                fig.savefig(os.getcwd() + "\\plot\\train_examples\\" + self.name[n][0] + "_" + self.name[n][1] + ".png",
                            dpi=300)
                fig.clf()
                plt.close()

    def set_parameter(self):
        # Convolution layers
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(6, (3, 3), activation='selu', padding='VALID', input_shape=(49, 49, 18)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(2, (3, 3), activation='selu', padding='VALID'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(1, (3, 3), activation='selu', padding='VALID'))

        # Fully-Connected Layer
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(2, activation='relu'))
        self.model.add(layers.Dense(1))
        self.model.summary()

    def fit(self, num_epoch):
        opt = tf.keras.optimizers.SGD(learning_rate=0.01)
        self.model.compile(optimizer=opt,
                           loss=tf.keras.losses.MSE,
                           metrics=['accuracy'])
        history = self.model.fit(self.train_input, self.train_label, batch_size=1, epochs=num_epoch,
                                 validation_data=(self.test_input, self.test_label))

        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.show()

        test_loss, test_acc = self.model.evaluate(self.test_input, self.test_label, verbose=2)

        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.show()

        print(test_acc)


if __name__ == "__main__":
    my_model = CNN()
    my_model.load_data(os.getcwd() + "\\data\\statistical_results\\info_list.pkl",
                       os.getcwd() + "\\data\\mapping_data\\")
    my_model.shuffle_data(plot=True)
    my_model.set_parameter()
    my_model.fit(10)
