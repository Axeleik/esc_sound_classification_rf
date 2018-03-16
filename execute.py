from functions import read_and_split_file,save_all_features_for_all_files,load_random_forest_classifier,\
    train_and_predict_with_rf,test_feature_importances,k_fold_cross_validation,k_fold_feature_exclusion,\
    eval_downwards_upwards,plot_eval_downwards_upwards



if __name__ == '__main__':

    initial_files_path="/mnt/localdata1/amatskev/esc_project_rf/sound_files/"
    split_files_path="/mnt/localdata1/amatskev/esc_project_rf/split_files/"
    path_for_whole_feature_set="/mnt/localdata1/amatskev/esc_project_rf/whole_feature_set.h5"
    path_for_whole_classes_set="/mnt/localdata1/amatskev/esc_project_rf/whole_classes_set.npy"
    rf_save_path="/mnt/localdata1/amatskev/esc_project_rf/rf_all_features.pkl"
    save_path_downwards="/mnt/localdata1/amatskev/esc_project_rf/downwards_results.pkl"
    save_path_upwards = "/mnt/localdata1/amatskev/esc_project_rf/upwards_results.pkl"

    # read_and_split_file(initial_files_path,split_files_path)

    features,classes=save_all_features_for_all_files(split_files_path,path_for_whole_feature_set,path_for_whole_classes_set)

    # rf=train_and_predict_with_rf(features,classes,[],rf_save_path,False)

    # test_feature_importances(rf)

    # k_fold_cross_validation(features,classes,10)

    # eval_downwards_upwards(features,classes,save_path_downwards,save_path_upwards)


    for fold in [2,5,10,20,30,40,50]:
        save_path_downwards = "/mnt/localdata1/amatskev/esc_project_rf/fold_eval/downwards_results_k_{}.pkl".format(fold)
        save_path_upwards = "/mnt/localdata1/amatskev/esc_project_rf/fold_eval/upwards_results_k_{}.pkl".format(fold)

        eval_downwards_upwards(features, classes, save_path_downwards, save_path_upwards,fold)
        # plot_eval_downwards_upwards(save_path_downwards,save_path_upwards,k_fold)


