##############################################
# Giotto Benchmarking - Kidney 10x Visium
# Authors: Chandrima Modak, Mauminah Raina
##############################################

#Load required packages
library(Matrix)
library(Seurat)
library(Giotto)
library(GiottoData)
library(dplyr)
library(readxl)
library(mclust)


# Filepaths ---------------------------------------------------------------
setwd("C:/Giotto_data/DATA_PART_7/")
getwd()
file_path=list.files(path="C:/BayesSpaces_data/DATA_PART_7/",pattern = "\\.RDS$")
print(file_path)
for (i in 1:length(file_path)){
  folder_name = sub("\\.RDS$", "",file_path[i])
  dir.create(folder_name)
}
filename = paste0("C:/BayesSpaces_data/DATA_PART_7/",c(list.files(path="C:/BayesSpaces_data/DATA_PART_7/",pattern = "\\.RDS$")))
file_path <- filename

getwd()

# Outputs -----------------------------------------------------------------
folders = list.dirs(path="C:/Giotto_data/DATA_PART_7/")
output = paste0(c(folders[-1]), '/')
print(output)


GiottoClusterFinder <- function(file_path,output){
  
  # The bayesSpaces block ---------------------------------------------------
  
  #Read the RDS file using the location from file_path
  sce <- readRDS(file_path)
  
  #Get the spatial coordinates
  coo_data <- sce@images$slice1@coordinates
  
  #Extract the counts data
  exp_data <- sce@assays[["Spatial"]]@counts
  
  #Filter the required coordinates
  coo_data <- coo_data[,c(-1,-4,-5)]
  
  #Extract the cell locations
  cell_ID <- rownames(coo_data)
  cell_ID <- data.frame(cell_ID)
  
  #Merge cell locations with their coordinates
  coo_data <- cbind(coo_data,cell_ID)
  
  #Assign column names to the coordinates data in the desired format
  colnames(coo_data) <- c("sdimx","sdimy","cell_ID")
  
  #Create a giotto object
  custom_giotto_object <- createGiottoObject(expression = exp_data,
                                             spatial_locs = coo_data)
  
  # Process the Giotto object, filtering, normalization, adding statistics and correcting for covariates
  custom_giotto_object <- processGiotto(custom_giotto_object,
                                        filter_params = list(expression_threshold = 0,
                                                             feat_det_in_min_cells = 10,
                                                             min_det_feats_per_cell = 10),
                                        norm_params = list(norm_methods = 'standard',
                                                           scale_feats = TRUE,
                                                           scalefactor = 6000),
                                        stat_params = list(expression_values = 'normalized'),
                                        adjust_params = list(expression_values = c('normalized'),
                                                             covariate_columns = 'nr_feats'))
  
  
  
  #Predict the highly variable features using loess regression
  custom_giotto_object <- calculateHVF(gobject = custom_giotto_object, method = 'cov_loess')
  
  
  
  
  ## Select genes highly variable genes that fit specified statistics
  # These are both found within feature metadata
  feature_metadata = getFeatureMetadata(custom_giotto_object)[]
  featgenes = feature_metadata[hvf == 'yes' & perc_cells > 4 & mean_expr_det > 0.5]$feat_ID
  
  ## run PCA on expression values (default)
  custom_giotto_object <- runPCA(gobject = custom_giotto_object, feats_to_use = featgenes, scale_unit = F, center = F)
  
  # # plot a scree plot
  # screePlot(custom_giotto_object)
  # 
  # 
  # # Plot a PCA
  # plotPCA(gobject = custom_giotto_object)
  
  #run tSNE
  custom_giotto_object <- runtSNE(custom_giotto_object, dimensions_to_use = 1:15)
  # plotTSNE(gobject = custom_giotto_object)
  
  #run UMAP
  custom_giotto_object <- runUMAP(custom_giotto_object, dimensions_to_use = 1:15)
  # plotUMAP(gobject = custom_giotto_object)
  
  #Create the nearest network
  custom_giotto_object <- createNearestNetwork(gobject = custom_giotto_object, type = "sNN", dimensions_to_use = 1:15, k = 15)
  
  ## k-means clustering
  custom_giotto_object <- doKmeans(gobject = custom_giotto_object, dim_reduction_to_use = 'pca')
  
  ## Leiden clustering - increase the resolution to increase the number of clusters
  custom_giotto_object <- doLeidenCluster(gobject = custom_giotto_object,
                                          resolution = 0.4,
                                          n_iterations = 1000,
                                          name = 'leiden_0.4_1000')
  
  ## Louvain clustering - increase the resolution to increase the number of clusters
  # The version argument may be changed to 'multinet' to run a Louvain algorithm
  # from the multinet package in R.
  custom_giotto_object <- doLouvainCluster(gobject = custom_giotto_object,
                                           version = 'community',
                                           resolution = 0.4)
  
  
  
  #Extract the clustering results and assign appropriate column names
  clustering_results <- data.frame(custom_giotto_object@cell_metadata[["cell"]][["rna"]]@metaDT[["cell_ID"]],custom_giotto_object@cell_metadata[["cell"]][["rna"]]@metaDT[["leiden_0.4_1000"]])
  
  colnames(clustering_results) <- c("Barcodes","Cluster")
  
  #Save the cluster results
  write.table(clustering_results, paste(output,'spatial_clustering_results.csv',sep = ""), sep=',',quote=FALSE,row.names = FALSE)
  
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
  write.table(new_t, paste(output[1],'labels.csv',sep = ""), sep=',',quote=FALSE)
  
  #return the adjusted rand index for this dataset
  return (adjustedRandIndex(custom_giotto_object@cell_metadata[["cell"]][["rna"]]@metaDT[["leiden_0.4_1000"]], new_t$class))
  
}
# GiottoClusterFinder(file_path,output)

#Create an empty dataframe to save ARI values
ARI_data <- data.frame()

#For loop to perform cluster analysis for each dataset
for(i in 11:length(file_path)){
  
  #Call the GiottoClusterFinder function with the RDS file path and output folder path
  ARI <- GiottoClusterFinder(file_path[i],output[i])
  
  #Save the ARI and the dataset As a new row
  new_row <- c(basename(file_path[i]),ARI)
  
  #Append the new row to the dataframe
  ARI_data <- rbind(ARI_data,new_row)
  
}



# Fixing_ARI_table --------------------------------------------------------

ARI_dataframe  <- data.frame()

ARI_fix = function(file_name){
  return(adjustedRandIndex(file_name,la))
}

#Edit the column names of the dataframe containing ARI values
colnames(ARI_data) <- c("Dataset","ARI")

#Write the dataframe to a csv file
write.table(ARI_data, paste('GiottoARI.csv',sep = ""), sep=',',quote=FALSE,row.names = FALSE)
