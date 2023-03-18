import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

import process_data

casualty_data = pd.read_csv('casualty_train.csv')
vehicle_data = pd.read_csv('vehicle_train.csv')

casualty_test = pd.read_csv('casualty_test.csv')
vehicle_test = pd.read_csv('vehicle_test.csv')



#process_data.process_data(casualty_data, vehicle_data)
#process_data.process_data(casualty_test, vehicle_test)

joined_data = pd.merge(casualty_data, vehicle_data, on = 'accident_reference', how='left')
test_join = pd.merge(casualty_test, vehicle_test, on='accident_reference', how='left')

joined_data["casualty_severity"] = joined_data['casualty_severity'].map({1:1,2:1,3:0})
test_join["casualty_severity"] = test_join['casualty_severity'].map({1:1,2:1,3:0})


#casualty_X = joined_data.drop(["casualty_severity", "accident_reference", "lsoa_of_casualty", "vehicle_reference_x", "vehicle_reference_y", "casualty_reference", "generic_make_model", "lsoa_of_driver", "sex_of_casualty", "age_band_of_casualty", "pedestrian_road_maintenance_worker", "casualty_imd_decile", "casualty_home_area_type", "vehicle_direction_from", "vehicle_direction_to", "sex_of_driver", "age_band_of_driver"], axis=1)
casualty_X = joined_data[["vehicle_location_restricted_lane", "hit_object_off_carriageway", "hit_object_in_carriageway", "vehicle_manoeuvre"]]

casualty_Y = joined_data[["casualty_severity"]].casualty_severity.values.tolist()




#x_train, x_test, y_train, y_test = train_test_split(casualty_X, casualty_Y, test_size=0.1, random_state=0)

x_train = casualty_X
y_train = casualty_Y

x_test = test_join[["vehicle_location_restricted_lane", "hit_object_off_carriageway", "hit_object_in_carriageway", "vehicle_manoeuvre"]]
#y_test = test_join[["casualty_severity"]].casualty_severity.values.tolist()

logisticRegr = LogisticRegression(max_iter=1000, class_weight= {0:100,1:200})
logisticRegr.fit(x_train, y_train)

result = logisticRegr.predict(x_test)


result = result.astype(int)

result = pd.DataFrame(result)
result.rename(columns = {0:'casualty_severity'}, inplace = True)

print(result)
result.to_csv('result.csv', index = False)







"""cm = confusion_matrix(y_test, result)

display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logisticRegr.classes_)

display.plot()

print(classification_report(y_test, result))
#plot_confusion_matrix(logisticRegr, x_test, y_test, cmap=plt.cm.Blues)

plt.show()"""


"""print(pd.DataFrame({
    'Predictions': result,
    'Actual Results': y_test,
    'INCORRECT?': abs(result - y_test)
}))"""
