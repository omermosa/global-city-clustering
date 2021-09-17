data=read.csv('Soc_econ_data_4paper.csv')

# head(data,5)

# dim(data)

require(robustX); library(robustbase);

X=as.matrix(data[,c(4,7)])


print(colMeans(X))

pairs(X)

output_X=mvBACON(X)

is_outlier_X=output_X$subset


X_nooutliers=subset(X, is_outlier_X)
X_outliers=subset(X, !is_outlier_X)

print(cor(X_nooutliers[,1],X_nooutliers[,2]))

print(cor(X_outliers[,1],X_outliers[,2]))

print(dim(X_outliers))

data_no=(subset(data, is_outlier_X))
data_o=(subset(data, !is_outlier_X))

plot(X_nooutliers)

d = dist(data_no[,2:10], method = "euclidean") 
hc = hclust(d, method="ward") 


plot(hc);



d = dist(data_o[,2:10], method = "euclidean") 
hc = hclust(d, method="ward") 
plot(hc)

plot(X_outliers)

write.csv(data_no,"all_data_noni_nooutliers_.csv", row.names = FALSE)


write.csv(data_o,"all_data_noni_outliers_.csv", row.names = FALSE)



