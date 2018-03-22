from functions import read_and_split_file,save_all_features_for_all_files,load_random_forest_classifier,\
    train_and_predict_with_rf,test_feature_importances,k_fold_cross_validation,k_fold_feature_exclusion,\
    eval_downwards_upwards,plot_eval_downwards_upwards,get_length_array


def plot_feature_importances():

    import numpy as np
    import matplotlib.pyplot as plt

    feature_importances = [
        ["chroma_stft", 0.0748233894585],
        ["chroma_cqt", 0.054921192219],
        ["chroma_cens", 0.0633787896738],
        ["malspectrogram", 0.5394277997],
        ["mfcc", 0.136398657179],
        ["rmse", 0.00674484749817],
        ["spectral_centroid", 0.00579915490682],
        ["spectral_bandwidth", 0.00794957999144],
        ["spectral_contrast", 0.0637908706312],
        ["spectral_flatness", 0.00508670589443],
        ["spectral_rolloff", 0.0050881935081],
        ["poly_features", 0.0107314061708],
        ["tonnetz", 0.0258594131688],
        ["zero_crossing_rate", 0.0]]

    feature_names=np.array([name for name,importance in feature_importances])
    feature_importances=np.array([importance for name,importance in feature_importances])

    sort_mask=np.argsort(feature_importances)[::-1]
    feature_names=feature_names[sort_mask]
    feature_importances=feature_importances[sort_mask]

    for idx,name in enumerate(feature_names):
        plt.bar(idx,feature_importances[idx],label=feature_names[idx])

    plt.xlabel('Feature',fontsize=30)
    plt.ylabel('Feature importance [%]',fontsize=30)
    plt.title("Feature importances given computed by the Random Forest",fontsize=30)
    plt.xticks([])
    plt.legend(fontsize="x-large")
    plt.tick_params("both",labelsize="x-large")
    plt.show()

def test_esc50(features,classes):


    for fold in [5,10]:

        for n_trees in [500]:

            print("Now computing for {} trees".format(n_trees))

            save_path_upwards = "/mnt/localdata1/amatskev/esc_project_rf/class_whole_eval/downwards_results_trees_{}_k_{}.pkl".format(n_trees,fold)
            save_path_downwards = "/mnt/localdata1/amatskev/esc_project_rf/class_whole_eval/upwards_results_trees_{}_k_{}.pkl".format(n_trees,fold)

            eval_downwards_upwards(features, classes, save_path_downwards, save_path_upwards,n_trees,fold)
            plot_eval_downwards_upwards(save_path_downwards,save_path_upwards,n_trees,fold,esc="ESC50")




    for class_idx in [0,40*10,40*20,40*30,40*40]:

        print("Now computing for class from {} to {}".format(class_idx,class_idx+40*10))

        for fold in [5,10]:

            for n_trees in [500]:

                print("Now computing for {} trees".format(n_trees))

                save_path_upwards = "/mnt/localdata1/amatskev/esc_project_rf/" \
                                      "class_whole_eval/downwards_results_trees_{}_cl_{}_k_{}.pkl".format(n_trees,class_idx,fold)
                save_path_downwards = "/mnt/localdata1/amatskev/esc_project_rf/" \
                                    "class_whole_eval/upwards_results_trees_{}_cl_{}_k_{}.pkl".format(n_trees,class_idx,fold)

                eval_downwards_upwards(features[class_idx:class_idx+40*10], classes[class_idx:class_idx+40*10],
                                       save_path_downwards, save_path_upwards, n_trees,fold)

                plot_eval_downwards_upwards(save_path_downwards, save_path_upwards, n_trees,fold,int(class_idx/400)+1,esc="ESC50")

def test_esc10(features,classes):
    import numpy as np

    esc_names=["Sneezing","Dog","Clocktick","Cryingbaby","Rooster","Rain","Seawaves","Cracklingfire","Helicopter","Chainsaw"]

    classes_for_extraction=[classes[idx][:-2] for idx in range(0,len(classes),40)]
    classes_for_extraction=np.concatenate([[cl]*40 for cl in classes_for_extraction])

    uniq,indexes_of_uniq=np.unique(classes_for_extraction,return_index=True)

    indexes_of_esc10=[np.where(uniq==name)[0][0] for name in esc_names]

    features_esc10=np.concatenate([features[index:index+40] for index in indexes_of_uniq[indexes_of_esc10]])
    classes_esc10 = np.concatenate([classes[index:index + 40] for index in indexes_of_uniq[indexes_of_esc10]])

    for fold in [5,10]:

        for n_trees in [500]:

            print("Now computing for {} trees".format(n_trees))

            save_path_upwards = "/mnt/localdata1/amatskev/esc_project_rf/esc10_eval/downwards_results_trees_{}_k_{}.pkl".format(n_trees,fold)
            save_path_downwards = "/mnt/localdata1/amatskev/esc_project_rf/esc10_eval/upwards_results_trees_{}_k_{}.pkl".format(n_trees,fold)

            eval_downwards_upwards(features_esc10, classes_esc10, save_path_downwards, save_path_upwards,n_trees,fold)
            plot_eval_downwards_upwards(save_path_downwards,save_path_upwards,n_trees,fold,esc="ESC10")

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

    # plot_feature_importances()

    # k_fold_cross_validation(features,classes,10)

    # eval_downwards_upwards(features,classes,save_path_downwards,save_path_upwards)

    test_esc50(features, classes)
    test_esc10(features, classes)
