# Load necessary libraries
library(readxl)
library(geosphere)

# Load the dataset (adjust the path if necessary)
file_path <- "Downloads/clinics.xls"  
df <- read_excel(file_path)

# Ensure latitude and longitude are numeric
df$locLat <- as.numeric(df$locLat)
df$locLong <- as.numeric(df$locLong)

# Display first few rows to check data types
str(df)

# Define the Haversine function
haversine <- function(lat1, lon1, lat2, lon2) {
  distHaversine(c(lon1, lat1), c(lon2, lat2)) / 1609.34  # Convert meters to miles
}

# Measure execution time for the for-loop approach
start_time <- Sys.time()
haversine_looping <- function(df) {
  distance_list <- numeric(nrow(df))
  for (i in 1:nrow(df)) {
    distance_list[i] <- haversine(40.671, -73.985, df$locLat[i], df$locLong[i])
  }
  return(distance_list)
}
df$distance_loop <- haversine_looping(df)
loop_time <- Sys.time() - start_time

# Measure execution time for mapply() approach (optimized apply)
start_time <- Sys.time()
df$distance_apply <- mapply(haversine, 40.671, -73.985, df$locLat, df$locLong)
apply_time <- Sys.time() - start_time

# Measure execution time for vectorized approach
start_time <- Sys.time()
df$distance_vectorized <- distHaversine(
  cbind(df$locLong, df$locLat), 
  c(-73.985, 40.671)
) / 1609.34  # Convert meters to miles
vectorized_time <- Sys.time() - start_time

# Tabulate the results
execution_times <- data.frame(
  Approach = c("For-loop", "Mapply", "Vectorized"),
  Execution_Time_Seconds = c(loop_time, apply_time, vectorized_time)
)

# Print results
print(execution_times)

