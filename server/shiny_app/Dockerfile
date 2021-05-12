FROM rocker/r-apt:bionic

RUN apt-get update && \
    apt-get install -y -qq \
    	r-cran-stopwords\
    	r-cran-shiny\
    	r-cran-httr\
    	r-cran-rjson\
    	r-cran-jsonlite\
    	r-cran-dplyr\
    	r-cran-stringr\
    	r-cran-tidytext\
    	r-cran-tidyr\
    	r-cran-ggplot2\
    	r-cran-plotly\
    	r-cran-shinythemes\
    	r-cran-lsa\
    	r-cran-scales\
    	r-cran-lubridate
RUN R -e 'install.packages("DT")'
RUN R -e 'install.packages("kableExtra")'
RUN R -e 'install.packages("Hmisc")'
COPY app.R app.R

CMD ["Rscript", "-e", "shiny::runApp('.', host = '0.0.0.0', port = 3838)"]