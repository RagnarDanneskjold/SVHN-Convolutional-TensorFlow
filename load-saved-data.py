#load the variables
def load_variables_2():
    house_number_y_train_load = pd.read_csv("house_number_y_train.csv")
    house_number_y_train_load = house_number_y_train_load.as_matrix()
    house_number_y_train_load = np.delete(house_number_y_train_load, 0, 1)
    print("house_number_y_train_load", house_number_y_train_load.shape)

    house_number_X_train_load = pd.read_csv("house_number_X_train.csv")
    house_number_X_train_load = house_number_X_train_load.as_matrix()
    house_number_X_train_load = np.delete(house_number_X_train_load, 0, 1)
    house_number_X_train_load = house_number_X_train_load.reshape(73257, 32, 32, 3)
    print("house_number_X_train_load", house_number_X_train_load.shape)

    house_number_y_test_load = pd.read_csv("house_number_y_test.csv")
    house_number_y_test_load = house_number_y_test_load.as_matrix()
    house_number_y_test_load = np.delete(house_number_y_test_load, 0, 1)
    print("house_number_y_test_load", house_number_y_test_load.shape)

    house_number_X_test_load = pd.read_csv("house_number_X_test.csv")
    house_number_X_test_load = house_number_X_test_load.as_matrix()
    house_number_X_test_load = np.delete(house_number_X_test_load, 0, 1)
    house_number_X_test_load = house_number_X_test_load.reshape(26032, 32, 32, 3)
    print("house_number_X_test_load", house_number_X_test_load.shape)

    train_house_sequence_load = pd.read_csv("train_house_sequence.csv")
    train_house_sequence_load = train_house_sequence_load.as_matrix()
    train_house_sequence_load = np.delete(train_house_sequence_load, 0, 1)
    train_house_sequence_load = train_house_sequence_load.reshape(33402, 50, 150, 3)
    print("train_house_sequence_load", train_house_sequence_load.shape)

    test_house_sequence_load = pd.read_csv("test_house_sequence.csv")
    test_house_sequence_load = test_house_sequence_load.as_matrix()
    test_house_sequence_load = np.delete(test_house_sequence_load, 0, 1)
    test_house_sequence_load = test_house_sequence_load.reshape(13068, 50, 150, 3)
    print("test_house_sequence_load", test_house_sequence_load.shape)

    train_house_sequence_y_num_load = pd.read_csv("train_house_sequence_y_num.csv")
    train_house_sequence_y_num_load = train_house_sequence_y_num_load.as_matrix()
    train_house_sequence_y_num_load = np.delete(train_house_sequence_y_num_load, 0, 1)
    print("train_house_sequence_y_num_load", train_house_sequence_y_num_load.shape)

    train_house_sequence_y_name_load = pd.read_csv("train_house_sequence_y_name.csv")
    train_house_sequence_y_name_load = train_house_sequence_y_name_load.as_matrix()
    train_house_sequence_y_name_load = np.delete(train_house_sequence_y_name_load, 0, 1)
    print("train_house_sequence_y_name_load", train_house_sequence_y_name_load.shape)

    train_house_sequence_y_label_load = pd.read_csv("train_house_sequence_y_label.csv")
    train_house_sequence_y_label_load = train_house_sequence_y_label_load.as_matrix()
    train_house_sequence_y_label_load = np.delete(train_house_sequence_y_label_load, 0, 1)
    print("train_house_sequence_y_label_load", train_house_sequence_y_label_load.shape)

    test_house_sequence_y_num_load = pd.read_csv("test_house_sequence_y_num.csv")
    test_house_sequence_y_num_load = test_house_sequence_y_num_load.as_matrix()
    test_house_sequence_y_num_load = np.delete(test_house_sequence_y_num_load, 0, 1)
    print("test_house_sequence_y_num_load", test_house_sequence_y_num_load.shape)

    test_house_sequence_y_name_load = pd.read_csv("test_house_sequence_y_name.csv")
    test_house_sequence_y_name_load = test_house_sequence_y_name_load.as_matrix()
    test_house_sequence_y_name_load = np.delete(test_house_sequence_y_name_load, 0, 1)
    print("test_house_sequence_y_name_load", test_house_sequence_y_name_load.shape)

    test_house_sequence_y_label_load = pd.read_csv("test_house_sequence_y_label.csv")
    test_house_sequence_y_label_load = test_house_sequence_y_label_load.as_matrix()
    test_house_sequence_y_label_load = np.delete(test_house_sequence_y_label_load, 0, 1)
    print("test_house_sequence_y_label_load", test_house_sequence_y_label_load.shape)
    
#load_variables_2()

print("Finished Loading")
