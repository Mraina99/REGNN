##############################################
# BayesSpace Benchmarking - Kidney 10x Visium
# Authors: Chandrima Modak, Mauminah Raina
##############################################

#Load required packages
library(Matrix)
library(BayesSpace)
library(Seurat)
library(SingleCellExperiment)
library(ggplot2)
library(BayesSpace)
library(dplyr)
library(readxl)


# Filepaths ---------------------------------------------------------------
setwd('C:/BayesSpaces_data/Dummy')
getwd()
file_path=list.files(path="C:/BayesSpaces_data/DATA_PART_7/",pattern = "\\.RDS$")
print(file_path)

# Outputs -----------------------------------------------------------------
output = c("C:/BayesSpaces_data/DATA_PART_7/analysis/V12U21-010_XY02_21-0069/",
           "C:/BayesSpaces_data/DATA_PART_7/analysis/V12U21-010_XY03_21-0070/",
           "C:/BayesSpaces_data/DATA_PART_7/analysis/V12U21-010_XY04_22-0087/",
           "C:/BayesSpaces_data/DATA_PART_7/analysis/V13J17-332_XY01_23-0119/",
           "C:/BayesSpaces_data/DATA_PART_7/analysis/V13J17-332_XY02_23-0122/",
           "C:/BayesSpaces_data/DATA_PART_7/analysis/V13J17-332_XY02_Ref/",
           "C:/BayesSpaces_data/DATA_PART_7/analysis/V13J17-332_XY03_23-0123/",
           "C:/BayesSpaces_data/DATA_PART_7/analysis/V13J17-332_XY04_23-0128/",
           "C:/BayesSpaces_data/DATA_PART_7/analysis/V19S25-016_XY01_18-0006/",
           "C:/BayesSpaces_data/DATA_PART_7/analysis/V19S25-017_XY03_13437/",
           "C:/BayesSpaces_data/DATA_PART_7/analysis/V19S25-019_XY02_M32/",
           "C:/BayesSpaces_data/DATA_PART_7/analysis/V19S25-019_XY03_M61/",
           "C:/BayesSpaces_data/DATA_PART_7/analysis/V19S25-019_XY04_F52/"
           )


bayesSpaceClusterFinder <- function(file_path,output){

# The bayesSpaces block ---------------------------------------------------

  
  #Read the RDS file using the location from file_path
  sce <- readRDS(file_path)
  
  #Get the spatial coordinates
  spa <- sce@images$slice1@coordinates
  
  #Convert the vaiable containing spatial coordinates into a dataframe
  colData <- data.frame(spa$row,spa$col,spa$imagerow,spa$imagecol)
  
  #Edit the column names of the dataframe containing spatial coordinates
  colnames(colData) <- c("row","col","imagerow","imagecol")
  
  #Construct a single cell experiment
  sce <- SingleCellExperiment(assays=list(counts=sce@assays[["Spatial"]]@data),
                              colData=colData)
  
  #Perform Spatial Preprocess
  set.seed(102)
  sce <- spatialPreprocess(sce, platform ='Visium')
  
  #qTune and qPlot to find the optimal number of clusters
  sce <- qTune(sce, qs=seq(2, 10), platform='Visium')
  # plot <- qPlot(sce)
  # print(plot)
  
  #Perform spatial cluster of the single cell experiment
  set.seed(149)
  sce <- spatialCluster(sce, q=4, platform='Visium',
                        init.method="mclust", model="t", gamma=2,
                        nrep=10000, burn.in=100,
                        save.chain=TRUE)
  
  # clusterPlot(sce)
  # 
  # clusterPlot(sce, palette=c("purple", "red", "blue", "yellow"), color="black") +
  #   theme_bw() +
  #   xlab("Column") +
  #   ylab("Row") +
  #   labs(fill="BayesSpace\ncluster")
  
  # sce.enhanced <- spatialEnhance(sce, q=qValue,nrep=1000, burn.in=100)
  # 
  # clusterPlot(sce.enhanced)
  
  #Extract the cluster results from the single cell experiment
  cluster_results <- data.frame(sce@colData@rownames,sce@colData@listData[["spatial.cluster"]])
  
  colnames(cluster_results) <- c("Barcodes","Cluster")
  
  #Save the cluster results
  write.table(cluster_results, paste(output,'spatial_clustering_results.csv',sep = ""), sep=',',quote=FALSE,row.names = FALSE)
  
  # Groundtruth block -------------------------------------------------------
  
  #Extract the ground truth labels
  cell_types = read_excel('cell_types.xlsx')
  print(colnames(cell_types))
  # Remove 'cluster column' and then remove duplicate rows
  cell_types <- cell_types[, !(names(cell_types) %in% c("cluster (v3)", "Subclass_Full"))] #remove cluster
  # cell_types = cell_types[,-1]
  # cell_types = cell_types[,-1] #remove subclass
  cell_types[2,1]
  cell_types = cell_types[!duplicated(cell_types),] #remove duplicates
  for (i in 1:nrow(cell_types)){
    if (cell_types[i,1] != cell_types[i,2]){
      cell_types[i,1] <- cell_types[i,2]
    }
  }
  cell_types = cell_types[!duplicated(cell_types),] #remove duplicate DCT
  #-----merge metadata df and cell_types using trasnfer_subset & subclass.l3------
  colnames(cell_types)[1] = "transfer_subset"
  ?left_join
  kidney1<-readRDS(file_path)
  t=kidney1@meta.data
  print(colnames(t))
  t <- tibble::rownames_to_column(t, "barcodes")
  if (!("transfer_subset" %in% colnames(t))){
    colnames(t)[which(names(t) == "subclass.l2")] <- "transfer_subset"
  }
  new_t = left_join(t[ ,c("barcodes", "transfer_subset")], cell_types[ , c("transfer_subset", "subclass.l1", 'class', 'substructure')], by="transfer_subset")
  if (anyNA(new_t)){
    print("NA detected in labels.csv")
    print(dataname)
    break
  }
  write.table(new_t, paste(output,'labels.csv',sep = ""), sep=',',quote=FALSE)
  
  #return the adjusted rand index for this dataset
  return(adjustedRandIndex(sce@colData$spatial.cluster, new_t$class))

}

#Create an empty dataframe to save ARI values
ARI_data <- data.frame()

#For loop to perform cluster analysis for each dataset
for(i in 1:length(file_path)){
  
  #Call the bayesSpaceClusterFinder function with the RDS file path and output folder path
  ARI <- bayesSpaceClusterFinder(file_path[i],output[i])
  
  #Save the ARI and the dataset As a new row
  new_row <- c(basename(file_path[i]),ARI)
  
  #Append the new row to the dataframe
  ARI_data <- rbind(ARI_data,new_row)
  
}

#Edit the column names of the dataframe containing ARI values
colnames(ARI_data) <- c("Dataset","ARI")

#Write the dataframe to a csv file
write.table(ARI_data, paste('BayesSpaceARI.csv',sep = ""), sep=',',quote=FALSE,row.names = FALSE)
