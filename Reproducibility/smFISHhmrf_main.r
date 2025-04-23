##############################################
# smFISHHMRF Benchmarking - Kidney 10x Visium
# Authors: Mauminah Raina
##############################################
library(Giotto)
library(smfishHmrf)

# Embeddings - Sample SSL57
counts <-read.csv(paste0("/N/slate/mraina/REGNN_final/REGNN_SSL/result/Embeddings/V10S14-085_XY04_21-0057_SSL_scGNN_logcpm_PEalpha3.0_zdim64_embedding.csv"), header = 0) 
counts <- data.frame(t(counts))

# Embeddings - Sample GAE57 
counts <-read.csv(paste0("/N/slate/mraina/REGNN_final/REGNN_GAE/result/Embeddings/V10S14-085_XY04_21-0057_GAE_scGNN_logcpm_PEalpha2.0_zdim8_embedding.csv"), header = 0) 
counts <- data.frame(t(counts))


View(t(counts))
perform = function(sample){
  folder <- paste0("/N/slate/mraina/Juexin/Eadon/newkidneydata/", sample, "/")
  counts <-read.csv(paste0(folder,"fishformat_count.txt"), sep = " ", header = FALSE, row.names = 1)
  spa <-read.csv(paste0(folder, "fishformat_spa.txt"), sep = " ", header = FALSE)
  spa$V1 <- NULL
  colnames(spa) <- c("x", "y")
  rownames(spa) <- colnames(counts)
  temp_dir <- paste0("/N/scratch/mraina/smFISHhmrf/GIotto/VisiumKidney23/", sample, "/") #/Kidney057_Emb

  
  
  seqfish_Vis <- createGiottoObject(raw_exprs = counts,
                                    spatial_locs = spa,
                                    instructions = myinstructions)
  
  # Preprocessing
  seqfish_mini <- filterGiotto(gobject = seqfish_Vis, 
                               expression_threshold = 0.5, 
                               gene_det_in_min_cells = 20, 
                               min_det_genes_per_cell = 0)
  seqfish_mini <- normalizeGiotto(gobject = seqfish_mini, scalefactor = 6000, verbose = T)
  seqfish_mini <- addStatistics(gobject = seqfish_mini)
  seqfish_mini <- adjustGiottoMatrix(gobject = seqfish_mini, 
                                     expression_values = c('normalized'),
                                     covariate_columns = c('nr_genes', 'total_expr'))
  seqfish_mini <- calculateHVG(gobject = seqfish_mini)
  seqfish_mini <- runPCA(gobject = seqfish_mini)

  
  # Making Spatial Grid [can start from here if using embedding]
  seqfish_mini <- createSpatialGrid(gobject = seqfish_mini, # change to seqfish_Vis when running embeddings
                                    sdimx_stepsize = 300,
                                    sdimy_stepsize = 300,
                                    minimum_padding = 50)
  #For embedding, using emdeddings as normalizsed expression
  # seqfish_mini@norm_expr <- as.matrix(counts)
  # seqfish_mini@norm_scaled_expr <- as.matrix(counts)
  
  # visualize grid
  seqfish_mini = createSpatialNetwork(gobject = seqfish_mini, minimum_k = 2, 
                                      maximum_distance_delaunay = 400)
  seqfish_mini = createSpatialNetwork(gobject = seqfish_mini, minimum_k = 2, 
                                      method = 'kNN', k = 10)

  
  # Spatial Genes
  km_spatialgenes = binSpect(seqfish_mini)
  
  # Spatial HMRF domains
  hmrf_folder = paste0(temp_dir,'HMRF/')
  if(!file.exists(hmrf_folder)) dir.create(hmrf_folder, recursive = T)
  
  # perform hmrf
  my_spatial_genes = km_spatialgenes[1:100]$genes
  
  # For embedding only
  # my_spatial_genes = seqfish_mini@gene_metadata$gene_ID
  
  HMRF_spatial_genes = doHMRF(gobject = seqfish_mini,
                              expression_values = 'scaled',
                              spatial_genes = my_spatial_genes,
                              spatial_network_name = 'Delaunay_network',
                              k = 4,
                              betas = c(28,2,2),
                              output_folder = paste0(hmrf_folder, '/', 'Spatial_genes/SG_top100_k4_scaled'))
  
  # Saving beta 28 version to object
  seqfish_mini = addHMRF(gobject = seqfish_mini,
                         HMRFoutput = HMRF_spatial_genes,
                         k = 4, betas_to_add = c(28),
                         hmrf_name = 'HMRF')
  
  # Extract Clusters
  seqfish_mini@cell_metadata$HMRF_k4_b.28
  outputclusters <- data.frame(seqfish_mini@cell_metadata$cell_ID, seqfish_mini@cell_metadata$HMRF_k4_b.28)
  colnames(outputclusters) <- c('IDs', 'Clusters')
  write.table(outputclusters,paste0(hmrf_folder, '/','hmrfclusters.csv'), sep = ",", row.names = FALSE, quote = FALSE)
  
}

sample_list = c("V10S14-085_XY01_20-0038",
                "V10S14-085_XY02_20-0040",
                "V10S14-085_XY03_21-0056",
                "V10S14-085_XY04_21-0057",
                "V10S14-086_XY01_21-0055",
                "V10S14-086_XY02_20-0039",
                "V10S14-086_XY03_21-0063",
                "V10S14-086_XY04_21-0066",
                "V10S14-087_XY01_21-0061",
                "V10S14-087_XY02_21-0063",
                "V10S14-087_XY03_21-0064",
                "V10S14-087_XY04_21-0065",
                "V10S15-102_XY02_IU-21-019-5",
                "V10S15-102_XY03_IU-21-015-2",
                "V10S21-388_XY01_21-0068",
                "V10S21-388_XY02_20-0071",
                "V10S21-388_XY03_20-0072",
                "V10S21-388_XY04_20-0073",
                "V19S25-016_XY01_18-0006",
                "V19S25-017_XY03_13437",
                "V19S25-019_XY02_M32",
                "V19S25-019_XY03_M61",
                "V19S25-019_XY04_F52"
)

myinstructions = createGiottoInstructions(save_dir = "/N/scratch/mraina/smFISHhmrf/GIotto/",
                                          save_plot = FALSE, 
                                          show_plot = FALSE)

# Use this loop to run the function!
for (i in 1:length(sample_list)){
  perform(sample_list[i])
  print(i)
}
