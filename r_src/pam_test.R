args = commandArgs(trailingOnly=TRUE)

if (! "cluster" %in% row.names(installed.packages()))
  install.packages("cluster")
library(cluster)

# check arguments
if (length(args) != 2){
  cat("Arguments error!\n")
  cat("Usage: <path to csv with data> <medoilds number>\n")
  quit(save = "no", status = 0)
}

data <- read.csv(file=args[1], sep = ",", header = F);
output <- cluster::pam(data, k=args[2], metric="euclidean")
write.csv(x=output$id.med, file = "medoids.csv", row.names = FALSE, )
write.csv(x=output$clustering, file = "labels.csv", row.names = FALSE, )
quit(save = "no", status = 0)
