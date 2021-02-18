# -*- coding: utf-8 -*-

"""
eng : Breast Cancer Classification --> The aim of the project, to determine whether the breast cancer cell is malignant or benign.
tr :  Göğüs Kanseri Sınıflandırma --> Projenin amacı göğüs kanseri tanısı konulan kişilerde, iyi huylu mu - kötü
huylu mu olduğunu tespit etmektedir.
"""

# eng : import library
# tr : Kütüphanelerimizi Projemize Dahil Ediyoruz.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler

# eng :  GridSearchCV --> We use it to find the best parameters.
# tr : GridSearchCV --> KNN best paremetleri bulmak için kullanırız.
from sklearn.model_selection import train_test_split, GridSearchCV 

# eng : to evaluate the result
# tr : Başarım Sonucu Değerlendirmek için Kullanırız 
from sklearn.metrics import accuracy_score, confusion_matrix 
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, LocalOutlierFactor
from sklearn.decomposition import PCA

# eng : Used to Ignore Non-Error Alerts
# tr :  Proje Sırasında Hata Olmayan Uyarıları Görmemek için Kullanırız
import warnings
warnings.filterwarnings("ignore")

# eng : load data
# tr : Verimizi okuyoruz.
data = pd.read_csv("cancer.csv")

# eng : We remove the features we do not want from our project (axis = 1 means drop column)
# tr : İstemediğimiz Features' ları projemizden çıkartıyoruz. (axis = 1 ' colunun drop edileceğini gösteriyor)
data.drop(['Unnamed: 32','id'], inplace = True, axis = 1)
 
# eng : We change the title of the properties
#  tr : Özelliklerin başlığını değiştiriyoruz
data = data.rename(columns = {"diagnosis":"target"})

# eng : WE START EXAMINING OUR DATA
# tr : VERİMİZİ İNCELEMEYE BASLIYORUZ

#eng : We are looking at how many Benign and Malignant cancer cells there are.
# tr : Kaç İyi Huylu, Kaç Kötü Huylu kanser hücresi olduğuna bakıyoruz.
sns.countplot(data["target"])
print(data.target.value_counts()) 

# eng : We convert string expressions to int because it will be necessary when making trains.(Bening = 0 , Malignant = 1)
# tr : String ifadeleri rakamlara çeviriyoruz çünkü train yaparken gerekli olucak.( Benign(İyi Huylu) = 0 , Malignant(Kötü Huylu) = 1 olarak değiştiriyoruz )
data["target"] = [1 if i.strip() == "M" else 0 for i in data.target] 

# eng : We learn the size of the data
# tr :  Datamızın boyutunu öğreniyoruz
print("Data Shape:", data.shape) 

# eng : we are reviewing our data
# tr : datamızı gözden geçiriyoruz
data.info() 

"""
eng : We look at the data need for standardization, if there are big differences between the data, standardization is required.
tr : Dataya bakıyoruz standardization gereklimi, eğer veriler arasında büyük farklar varsa standardization gereklidir.
"""
describe = data.describe()

# eng : We perform our detailed data analysis.
# tr : Detaylı veri analizimizi gerçekleştiriyoruz.
corr_matrix = data.corr()

#eng : We look at the relationship between features.
# tr : özellikler arasındaki ilişkiye bakıyoruz.
sns.clustermap(corr_matrix, annot = True, fmt = ".2f")
plt.title("Correlation Between Features")
plt.show()

"""
eng : First, we set a limit value. Here we set it to 0.75.
We bring the ones whose relationship between properties is greater than 0.75.
tr : ilk olarak sınır değer belirliyoruz. Burada 0.75 olarak belirledik. 
Özellikler arasındaki ilişki 0.75 den büyük olanları getiriyoruz.
"""

threshold = 0.75 
filtre = np.abs(corr_matrix["target"]) > threshold 
corr_features = corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_features].corr(), annot = True, fmt = ".2f")
plt.title("Correlation Between Features w Corr Theshold 0.75")
plt.show()

"""
eng : pd.melt --> It allows us to dissolve the data we have, that is, it helps us to collect the columns we want in a single column.
      id_vars-->  We write here which columns we want.
tr : pd.melt -->  elimizdeki datayı eritmemizi sağlar, yani istediğimiz kolonları tek bir kolonda toplamamıza yarar.
     id_vars--> Hangi kolonları istediğimizi buraya yazarız.
"""
data_melted = pd.melt(data, id_vars = "target",
                      var_name = "features",
                      value_name = "value")

"""
eng : plt.boxplot --> This type of chart shows the quartile values ​​of the distribution with outliers.
tr : plt.boxplot --> Bu tür bir grafik, dağıtımın çeyrek değerlerini-aşırı değerler ile birlikte göstermektedir.
"""
plt.figure()
sns.boxplot(x = "features", y = "value", hue = "target", data = data_melted)
plt.xticks(rotation = 90) 
plt.show()

"""
eng : sns.pairplot --> Draws bidirectional relationships for numerical columns throughout the entire data environment.
tr : sns.pairplot --> Tüm veri çevresi boyunca, sayısal sütünlar için çift yönlü ilişkiler çizer.
"""
sns.pairplot(data[corr_features], diag_kind = "kde", markers = "+", hue = "target")
plt.show()


"""
eng : We divide the data into two parts, y and x. (y --> Target , x --> all columns except target )
tr : veriyi y ve x olmak üzere iki parçaya ayırıyoruz. (y --> Target , x --> target hariç tüm kolonlar )
"""
y = data.target
x = data.drop(["target"], axis = 1)


columns = x.columns.tolist()

"""
eng : It is used to detect outliers.
tr : aykırı değerleri tespit etmek için kullanılır.
"""
# tr : outlier -->  Datasetimiz içerisinde diğer verilerden daha farklı aykırı olan veriler.
clf = LocalOutlierFactor()

# eng : We see if it is outlier or not, -1 is outlier
# tr : outlier mi değilmi diye bakıyoruz -1 ise outlier
y_pred = clf.fit_predict(x)

# eng : we need outlier factor values(From here we can better observe the outlier ones)
# tr : outlier factor değerlere ihtiyacımız var(Buradan outlier olanları daha iyi gözlemleyebiliriz)
X_score = clf.negative_outlier_factor_

# eng : We create dataframe and put outlier factor values ​​here
# tr : dataframe oluşturup outlier factor değerlerini buraya atıyoruz
outlier_score = pd.DataFrame()
outlier_score["score"] = X_score

# eng : will show us those with outlier value above 2.5
# tr : outlier değeri 2,5 üstü olanları bize göstericek
threshold = -2.5
filtre = outlier_score["score"] < threshold
outlier_index = outlier_score[filtre].index.tolist()

# eng : we take a look at the distribution of data
# tr : verinin dağılımına göz atıyoruz
plt.figure()
plt.scatter(x.iloc[outlier_index,0],x.iloc[outlier_index,1],color = "blue", s = 50, label = "outliers")
plt.scatter(x.iloc[:,0],x.iloc[:,1],color = "k", s = 3, label = "Data Points")

# eng : We do normalization for the plotting process
# tr : çizdirme işlemi için normalizasyon yapıyoruz
radius = (X_score.max()- X_score) / (X_score.max() - X_score.min())
outlier_score["radius"] = radius

# eng : we take a look at the diameter of cancer cells
# tr : kanser hücrelerinin çaplarına göz atıyoruz
plt.scatter(x.iloc[:,0],x.iloc[:,1],s = 1000*radius, edgecolors = "r", facecolors = "none", label = "Outlier Scores")
plt.legend() 
plt.show()


# eng : drop outlier
# tr : outlier drop ediyoruz
x = x.drop(outlier_index)
y = y.drop(outlier_index).values

#eng : Train test split --> We separate our test and training data
# tr :  Train test split -->  Test ve eğitim verilerimizi ayırıyoruz
test_size = 0.3
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = test_size, random_state = 42)

# standrization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 

X_train_df = pd.DataFrame(X_train, columns = columns)
X_train_df_describe = X_train_df.describe()
X_train_df["target"] = Y_train
 
"""
eng : pd.melt --> It allows us to dissolve the data we have, that is, it helps us to collect the columns we want in a single column.
      id_vars-->  We write here which columns we want.
tr : pd.melt -->  elimizdeki datayı eritmemizi sağlar, yani istediğimiz kolonları tek bir kolonda toplamamıza yarar.
     id_vars--> Hangi kolonları istediğimizi buraya yazarız.
"""
data_melted = pd.melt(X_train_df, id_vars = "target",
                      var_name = "features",
                      value_name = "value")

# eng : We examine our data by visualizing
# tr : görselleştirerek verimizi inceliyoruz
plt.figure()
sns.boxplot(x = "features", y = "value", hue = "target", data = data_melted)
plt.xticks(rotation = 90)
plt.show()
 
sns.pairplot(X_train_df[corr_features], diag_kind = "kde", markers = "+",hue = "target")
plt.show()

"""
eng : We include the knn classification algorithm we will use in our project.
tr : Kullanacağımız knn sınıflandırma algoritmasını projemize dahil ediyoruz.
"""

# eng :We determine the number of neighbors as 2 and create our object.
# tr : komşu sayını 2 olarak belirliyoruz ve nesnemizi oluşturuyoruz.
knn = KNeighborsClassifier(n_neighbors = 2)

# eng : We train our algorithm.
# tr : Algoritmamızı eğitiyoruz.
knn.fit(X_train, Y_train)

# eng : Now that our model is ready, it makes a predict.
# tr : artık modelimiz hazır olduğuna göre bir tahminde bulunuyoruz.
y_pred = knn.predict(X_test)

# eng : We are looking at how right and how wrong we predict.
# tr : Ne kadar doğru, ne kadar yanlış tahmin de bulunmuşuz buna bakıyoruz.
cm = confusion_matrix(Y_test, y_pred)

# eng : We look at our success score.
# tr : Başarı skorumuza bakıyoruz.
acc = accuracy_score(Y_test, y_pred)
score = knn.score(X_test, Y_test)

print("Score : ", score)
print("CM : ", cm)
print("Basic KNN Acc : ", acc)

# eng : We need to determine the best parameters. We determine the best parameters for our project using the function below.
# tr : best paremetreleri belirlememiz lazım. Projemiz için en iyi paremetreleri aşağıdaki fonksiyonu kullanarak tespit ediyoruz.
def KNN_Best_Params(x_train, x_test, y_train, y_test):
    
    # eng : We write the parameters to be tested for the knn algorithm. We will find the best among these parameters.
    # tr : knn algoritması için denenecek parametreleri yazıyoruz. Bu parametreler arsında en iyi olanı bulacağız.
    k_range = list(range(1,31))
    weight_options = ["uniform","distance"]
    print()
    param_grid = dict(n_neighbors = k_range, weights = weight_options)
    
    
    knn = KNeighborsClassifier()
    
    # eng : Using GridSearchCV we find our best parameters.
    # tr : GridSearchCV ' yi kullanarak en iyi parametrelerimizi buluyoruz.
    grid = GridSearchCV(knn, param_grid, cv = 10, scoring = "accuracy")
    grid.fit(x_train, y_train)
    
    # eng : grid.best_score_ -->  gives the best score. grid.best_params_ --> gives the parameters returning the best score.
    # tr : grid.best_score_ --> en iyi skoru verir. grid.best_params_ --> en iyi skoru döndüren parametreleri verir.
    print("Best training score : {} with paremeters : {}".format(grid.best_score_, grid.best_params_))
    print()
    
    knn = KNeighborsClassifier(**grid.best_params_) # best paremetre olarak gelen değerlerimiz.
    knn.fit(x_train, y_train)
    
    y_pred_test = knn.predict(x_test)
    y_pred_train = knn.predict(x_train)
    
    cm_test = confusion_matrix(y_test, y_pred_test)
    cm_train = confusion_matrix(y_train, y_pred_train)
    
    acc_test = accuracy_score(y_test, y_pred_test) 
    acc_train = accuracy_score(y_train, y_pred_train)
    print("Test Score: {}, Train Score: {}".format(acc_test, acc_train))
    print()
    print("CM Test: ",cm_test)
    print("CM Train: ",cm_train)
    
    return grid


# eng : We send our data by calling the function we wrote.
#  tr: yazdığımız fonksiyonu çağırarak verilerimizi gönderiyoruz.
grid = KNN_Best_Params(X_train, X_test, Y_train, Y_test) 


# PCA
"""
eng : PCA -- > The method that reduces the size of the data by keeping as much information as possible. We can reduce certain features.
tr : PCA --> Kısaca açıklamak gerekirse, mümkün olduğa kadar bilgi tutarak verinin boyutunu azaltan yöntem.Belli 
başlı özellikleri azaltabiliriz. 
"""
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# eng : We reduce 30 featurs to 2 
 # tr : 30 tane olan featurları 2 ye indiriyoruz
pca = PCA(n_components = 2)
pca.fit(x_scaled)
X_reduced_pca = pca.transform(x_scaled)
pca_data = pd.DataFrame(X_reduced_pca, columns = ["p1","p2"])
pca_data["target"] = y


sns.scatterplot(x = "p1", y = "p2", hue = "target", data = pca_data)
plt.title("PCA : p1 vs p2")

# eng : we are retraining now that we have a new dataset
# tr : artık yeni bir veri setimiz olduğu için tekrar eğitiyoruz
X_train_pca, X_test_pca, Y_train_pca, Y_test_pca = train_test_split(X_reduced_pca, y, test_size = test_size, random_state = 42)

# eng : We send our data by calling the function we wrote.
#  tr: yazdığımız fonksiyonu çağırarak verilerimizi gönderiyoruz.
grid_pca = KNN_Best_Params(X_train_pca, X_test_pca, Y_train_pca, Y_test_pca)

# eng : We are doing the visualization process here. We choose 4 colors
# tr : Burada görselleştirme işlemi yapıyoruz. 4 tane renk seçiyoruz 
cmap_light = ListedColormap(['orange',  'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'darkblue'])

h = .05 # step size in the mesh
X = X_reduced_pca
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = grid_pca.predict(np.c_[xx.ravel(), yy.ravel()])


# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
            edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("%i-Class classification (k = %i, weights = '%s')"
          % (len(np.unique(y)),grid_pca.best_estimator_.n_neighbors, grid_pca.best_estimator_.weights))

# NCA
"""
eng : NCA -- > Linear changes of input data to maximize classification performance.
learning the distance metric using the transformation.

tr : NCA --> Amacı leave one out sınıflandırma performansını maksimize edecek şekilde input verilerinin doğrusal
dönüşümünü kullanarak mesafe metriğini öğrenmektir.
Leave one out : Belirli bir mesafe ölçüsü kullanarak K en yakın komşunun birlikte başka tek bir noktayı predict 
etmeye çalıştığı yöntemdir.
"""

nca = NeighborhoodComponentsAnalysis(n_components = 2, random_state = 42)
nca.fit(x_scaled, y)
X_reduced_nca = nca.transform(x_scaled)
nca_data = pd.DataFrame(X_reduced_nca, columns = ["p1","p2"])
nca_data["target"] = y
sns.scatterplot(x = "p1",  y = "p2", hue = "target", data = nca_data)
plt.title("NCA: p1 vs p2")

X_train_nca, X_test_nca, Y_train_nca, Y_test_nca = train_test_split(X_reduced_nca, y, test_size = test_size, random_state = 42)

grid_nca = KNN_Best_Params(X_train_nca, X_test_nca, Y_train_nca, Y_test_nca)

cmap_light = ListedColormap(['orange',  'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'darkblue'])

h = .2 # step size in the mesh
X = X_reduced_nca
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = grid_nca.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
            edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("%i-Class classification (k = %i, weights = '%s')"
          % (len(np.unique(y)),grid_nca.best_estimator_.n_neighbors, grid_nca.best_estimator_.weights))

# eng : we find the wrong classifications we made.
# tr : yaptığımız yanlış sınıflanıdırmları buluyoruz.
knn = KNeighborsClassifier(**grid_nca.best_params_)
knn.fit(X_train_nca,Y_train_nca)
y_pred_nca = knn.predict(X_test_nca)
acc_test_nca = accuracy_score(y_pred_nca,Y_test_nca)
knn.score(X_test_nca,Y_test_nca)


test_data = pd.DataFrame()
test_data["X_test_nca_p1"] = X_test_nca[:,0]
test_data["X_test_nca_p2"] = X_test_nca[:,1]
test_data["y_pred_nca"] = y_pred_nca
test_data["Y_test_nca"] = Y_test_nca

plt.figure()
diff = np.where(y_pred_nca!=Y_test_nca)[0]
plt.scatter(test_data.iloc[diff,0],test_data.iloc[diff,1],label = "Wrong Classified",alpha = 0.2,color = "red",s = 1000)

sns.scatterplot(x="X_test_nca_p1", y="X_test_nca_p2", hue="Y_test_nca",data=test_data)




























