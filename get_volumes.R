options(repos = c(CRAN = "https://cloud.r-project.org"))


library(iRfcb)

base_dir <- '/proj/berzelius-2023-48/ifcb/main_folder_karin/data/tangesund_ifcb_raw_by_depth'
hdr_files <- list.files(base_dir, pattern = '.hdr$', full.names = TRUE, recursive = TRUE)

df_to_write <- data.frame()

for (hdr_file in hdr_files) {

    #check if file is a copy (has (1) in the name)
    if (grepl('\\(1\\)', hdr_file) || grepl('\\(2\\)', hdr_file)) {
        next
    }

    sample_name = gsub('.hdr', '', hdr_file)
    # take what is after the last slash
    sample_name = gsub('.*/', '', sample_name)
    print(sample_name)



    run_time = ifcb_get_runtime(hdr_file)
    print(run_time)

    if (run_time[1] <0.06) {
        print(paste('run time is less than 0.06 for', sample_name))
        next
    }
    
    volume_analyzed <- ifcb_volume_analyzed(hdr_file)
    print(volume_analyzed)

    df_to_write <- rbind(df_to_write, data.frame(sample_name = sample_name, volume_analyzed = volume_analyzed))
}

# save df

write.csv(df_to_write, file = '/proj/berzelius-2023-48/ifcb/main_folder_karin/data/tangesund_ifcb_raw_by_depth/volumes.csv', row.names = TRUE)