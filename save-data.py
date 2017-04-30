#save the variables
def save_variables_2():
    print("house_number_y_train: ", house_number_y_train.shape, type(house_number_y_train))
    house_number_y_train_dataframe = pd.DataFrame(house_number_y_train)
    house_number_y_train_dataframe.to_csv("house_number_y_train.csv")

    print("house_number_X_train: ", house_number_X_train.shape, type(house_number_X_train))
    house_number_X_train = house_number_X_train.ravel()
    house_number_X_train_dataframe = pd.DataFrame(house_number_X_train)
    house_number_X_train_dataframe.to_csv("house_number_X_train.csv")

    print("house_number_y_test: ", house_number_y_test.shape, type(house_number_y_test))
    house_number_y_test_dataframe = pd.DataFrame(house_number_y_test)
    house_number_y_test_dataframe.to_csv("house_number_y_test.csv")

    print("house_number_X_test: ", house_number_X_test.shape, type(house_number_X_test))
    house_number_X_test = house_number_X_test.ravel()
    house_number_X_test_dataframe = pd.DataFrame(house_number_X_test)
    house_number_X_test_dataframe.to_csv("house_number_X_test.csv")

    print("train_house_sequence: ", train_house_sequence.shape, type(train_house_sequence))
    train_house_sequence = train_house_sequence.ravel()
    train_house_sequence_dataframe = pd.DataFrame(train_house_sequence)
    train_house_sequence_dataframe.to_csv("train_house_sequence.csv")

    print("test_house_sequence: ", test_house_sequence.shape, type(test_house_sequence))
    test_house_sequence = test_house_sequence.ravel()
    test_house_sequence_dataframe = pd.DataFrame(test_house_sequence)
    test_house_sequence_dataframe.to_csv("test_house_sequence.csv")

    print("train_house_sequence_y_num: ", train_house_sequence_y_num.shape, type(train_house_sequence_y_num))
    train_house_sequence_y_num_dataframe = pd.DataFrame(train_house_sequence_y_num)
    train_house_sequence_y_num_dataframe.to_csv("train_house_sequence_y_num.csv")

    print("train_house_sequence_y_name: ", train_house_sequence_y_name.shape, type(train_house_sequence_y_name))
    train_house_sequence_y_name_dataframe = pd.DataFrame(train_house_sequence_y_name)
    train_house_sequence_y_name_dataframe.to_csv("train_house_sequence_y_name.csv")

    print("train_house_sequence_y_label: ", train_house_sequence_y_label.shape, type(train_house_sequence_y_label))
    train_house_sequence_y_label_dataframe = pd.DataFrame(train_house_sequence_y_label)
    train_house_sequence_y_label_dataframe.to_csv("train_house_sequence_y_label.csv")

    print("test_house_sequence_y_num: ", test_house_sequence_y_num.shape, type(test_house_sequence_y_num))
    test_house_sequence_y_num_dataframe = pd.DataFrame(test_house_sequence_y_num)
    test_house_sequence_y_num_dataframe.to_csv("test_house_sequence_y_num.csv")

    print("test_house_sequence_y_name: ", test_house_sequence_y_name.shape, type(test_house_sequence_y_name))
    test_house_sequence_y_name_dataframe = pd.DataFrame(test_house_sequence_y_name)
    test_house_sequence_y_name_dataframe.to_csv("test_house_sequence_y_name.csv")

    print("test_house_sequence_y_label: ", test_house_sequence_y_label.shape, type(test_house_sequence_y_label))
    test_house_sequence_y_label_dataframe = pd.DataFrame(test_house_sequence_y_label)
    test_house_sequence_y_label_dataframe.to_csv("test_house_sequence_y_label.csv")
    
#save_variables_2() #run this code to save variables as csv files

print("Finished saving")
