from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, roc_curve, auc
from statistics import mean
import pandas as pd
import numpy as np
import os, time, logging, cv2
import matplotlib.pyplot as plt
import seaborn as sns
import gradcam, miscellaneous
import config as c

def k_fold_test(name, k_fold, test_all=False, grad_cam=False, show_summary=False):
    """ Computes the precitions of k-fold models.
        Saves the test results (AUROC, accuracies, sensitivies, ... etc.).

        Args:
            name: the name of the model.
            k_fold: nth k-fold.
            test_all: it decides whether it will iterate through every k-fold model for testing or only use a nth k-fold model for testing.
            grad_cam: it decides whether Grad-CAMs of the testset are created or not. 
    """
    if not os.path.isdir(c.RESULT_PATH):
        os.mkdir(c.RESULT_PATH)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    fh = logging.FileHandler(os.path.join(c.RESULT_PATH, 'test_results.log'))
    fh.setLevel(level=logging.INFO)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    k_folds = []
    accuracies = []
    specificities = []
    sensitivities = []
    negative_predict_values = []
    precisions = []
    f1_scores = []
    aurocs = []

    plt.figure(num=1, figsize=(10, 10))
    plt.title('Receiver Operating Characteristic')
    colors = ['lightcoral', 'firebrick', 'chocolate', 'brown', 'orange', 'olive', 'green', 'teal', 'midnightblue', 'purple']

    if test_all:
        start_k = 1
        end_k = k_fold + 1
        roc_graph_file = 'roc_graph_all' + c.IMG_FORMAT
    else:
        start_k = k_fold
        end_k = k_fold + 1
        roc_graph_file = 'roc_graph_%d'%k_fold + c.IMG_FORMAT
    roc_graph_path = os.path.join(c.RESULT_PATH, roc_graph_file)

    for i in range(start_k, end_k):

        test_datagen = ImageDataGenerator(rescale=c.RESCALE)

        new_name = name + '_' + str(i)
        logger.info('- Testing K-fold = %d'%i)
        logger.info('- Model name: %s'%new_name)
        
        model = miscellaneous.get_model(new_name, custom_objects=c.CUSTOM_OBJECTS, show_summary=show_summary)
        model.save(os.path.join(c.MODEL_PATH, name + c.MODEL_FORMAT))
        
        test_generator = test_datagen.flow_from_directory(directory=c.TEST_PATH,
                                                          target_size=c.TARGET_SIZE,
                                                          batch_size=1,
                                                          color_mode=c.COLOR_MODE,
                                                          shuffle=c.TEST_SHUFFLE,
                                                          class_mode=c.CLASS_MODE)
        filenames = test_generator.filenames
        ids = []
        ground_truth = []
        for filename in filenames:
            filename = filename.split('/')[-1].split('.')[0]
            label = filename.split('_')[-1]
            img_id = filename.split('_')[0] + '/' + filename.split('_')[1] + c.IMG_FORMAT
            ids.append(img_id)
            ground_truth.append(float(label))

        intial_time = time.time()
        logger.info('- Prediction has started')
        pred = model.predict_generator(test_generator, steps=test_generator.n)
        pred_time = time.time() - intial_time

        pred = np.concatenate(pred).ravel()

        ground_truth = np.array(ground_truth)
        fpr, tpr, thresholds = roc_curve(ground_truth, pred, pos_label=1)
        auroc = auc(fpr, tpr)
        graph_label = 'ROC fold %d (AUC = %0.2f)' % (i, auroc)
        plt.plot(fpr, tpr, 'b', color=colors[i-1], label=graph_label)

        pred[pred < c.THRESHOLD] = 0
        pred[pred >= c.THRESHOLD] = 1

        df_pred = pd.DataFrame({'ID': ids, 'Ground Truth': ground_truth, 'Prediction': pred})
        df_pred_file = 'prediction_%d'%i + c.CSV_FORMAT
        df_pred.to_csv(os.path.join(c.RESULT_PATH, df_pred_file), index=False)
        logger.info('- Prediction has been completed')
        logger.info('- Prediction time taken: %f seconds'%pred_time)
        logger.info('- Prediction result has been saved as \'%s\' in \'results\' folder'%df_pred_file)

        plt.figure(num=2, figsize=(5,5))
        plt.title('K-fold = %d'%i)
        pd_gt = pd.Series(ground_truth, name='Actual')
        pd_pred = pd.Series(pred, name='Predicted')
        pd_cm = pd.crosstab(pd_gt, pd_pred, rownames=['Actual'], colnames=['Predicted'])
        logger.info('- Confusion matrix has been created')
        logger.info(pd_cm)
        cm_categories = ['Benign', 'Malign']
        sns.heatmap(pd_cm, annot=True, fmt=',', cbar=False,
                    xticklabels=cm_categories, yticklabels=cm_categories, cmap='Blues')
        cm_img_file = 'confusion_matrix_%d'%i + c.IMG_FORMAT
        
        plt.savefig(os.path.join(c.RESULT_PATH, cm_img_file))
        plt.close(2)
        logger.info('- Confusion matrix image has been saved as \'%s\' in \'results\' folder'%cm_img_file)

        # Note that in binary classification, 
        # recall of the positive class is also known as “sensitivity”; recall of the negative class is “specificity”.
        result_dict = classification_report(ground_truth, pred,
                                            target_names=c.LABELS,
                                            output_dict=c.OUTPUT_DICT)
        print(result_dict)
        k_folds.append(str(i))
        accuracies.append(result_dict['accuracy'])
        sensitivities.append(result_dict['malign']['recall'])
        specificities.append(result_dict['benign']['recall'])
        precisions.append(result_dict['malign']['precision'])
        negative_predict_values.append(result_dict['benign']['precision'])
        f1_scores.append(result_dict['malign']['f1-score'])
        aurocs.append(auroc)
       
        df_result = pd.DataFrame(
            {'K-Fold': [k_folds[-1]],
             'Accuracy': [accuracies[-1]],
             'Specificity': [specificities[-1]],
             'Sensitivity': [sensitivities[-1]],
             'Negative Predict Value': [negative_predict_values[-1]],
             'Precision': [precisions[-1]],
             'F1-score': [f1_scores[-1]],
             'AUROC': [aurocs[-1]]
             })
        logger.info(df_result)
        result_file = 'test_result_%d'%i + c.CSV_FORMAT
        result_path = os.path.join(c.RESULT_PATH, result_file)
        df_result.to_csv(result_path, index=False)
        logger.info('- This result has been saved as \'%s\' in \'results\' folder'%result_file)

        if grad_cam:
            if not os.path.isdir(c.HEATMAP_PATH):
                os.mkdir(c.HEATMAP_PATH)
            
            grad_cam_count = 10
            test_imgs = os.listdir(c.TEST_PATH + '/all')
            import random
            random.shuffle(test_imgs)
            test_imgs = test_imgs[:grad_cam_count]

            for test_img in test_imgs:
                img_arr = cv2.imread(os.path.join(c.TEST_PATH + '/all', test_img))
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
                img_id = test_img.split('.')[0]
                # actual_label = test_img.split('.')[0].split('_')[-1]
                heatmap_overlay, pred, pred_label = gradcam.get_heatmap_overlay(img_arr=img_arr, model=model)
                heatmap_file = '%s_%d_%d'%(img_id,pred_label,i) +c.IMG_FORMAT
                miscellaneous.save_img(img_arr=heatmap_overlay,
                                    img_path=os.path.join(c.HEATMAP_PATH, heatmap_file))
                logger.info('- Grad-CAM image has been saved as \'%s\' in \'heatmaps\' folder'%heatmap_file)

    x_min, x_max = 0, 0.3
    y_min, y_max = 0.7, 1
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(roc_graph_path)
    plt.close(1)
    logger.info('- ROC graph has been saved as \'%s\' in \'results\' folder'%roc_graph_file)

    if k_fold == c.K_FOLD and test_all:
        result_all_file = 'test_result_all' + c.CSV_FORMAT
        result_all_path = os.path.join(c.RESULT_PATH, result_all_file)

        k_folds.append('Average')
        for l in [accuracies, specificities, sensitivities, negative_predict_values, precisions, f1_scores, aurocs]:
            l.append(mean(l))
        
        df_total_result = pd.DataFrame(
            {'K-Fold': k_folds,
             'Accuracy': accuracies,
             'Specificity': specificities,
             'Sensitivity': sensitivities,
             'Negative Predict Value': negative_predict_values,
             'Precision': precisions,
             'F1-score': f1_scores,
             'AUROC': aurocs
             })
        logger.info(df_total_result)
        df_total_result.to_csv(result_all_path, index=False)
        logger.info('- This total result has been saved as \'%s\' in \'results\' folder'%result_all_file)