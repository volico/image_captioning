#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)
library(httr)
library(rjson)
library(jsonlite)
library(dplyr)
library(stringr)
library(tidytext)
library(stopwords)
library(tidyr)
library(ggplot2)#systemfonts svglite
library(plotly)
library(DT)
library(shinythemes)


# Define UI for application that draws a histogram
ui <- tagList(
  fluidPage(
    theme=shinytheme("superhero"),
    # Application title
    titlePanel("Image Captioning"),
    
    # Sidebar with a slider input for number of bins
    sidebarLayout(
      sidebarPanel(
        fileInput("file", label = h3("File input")),
        sliderInput("diff_value", "Establish the coefficient of similarity", 0, 1, 1),
        submitButton("Start the analysis")
      ),
      
      
      # Show a plot of the generated distribution
      mainPanel(
        dataTableOutput("interestingTable"),
        
        plotlyOutput("interestingPlot")
      )    
      
    )
  ),
  shinythemes::themeSelector())

# Define server logic required to draw a histogram
server <- function(input, output) {
  
  CaptData <- reactive({
    x = POST("http://94.19.209.176:8888/get_video_captions", body = (list(video = upload_file(input$file$datapath))))
    y = fromJSON(rawToChar(x$content)) #/Users/simon/Desktop/for_shiny/Cup.mp4
    caption <- y$captions
    data <- data.frame(caption, stringsAsFactors = FALSE)
    data$id <- as.numeric(rownames(data))
    data
    
  })
  
  CaptDataFull <- reactive({
    caption_data <- CaptData()
    caption_text <- caption_data
    
    means_of_embedded_words = matrix(0, nrow = length(y$embedded_words), ncol = 512)
    
    for (x in 1:length(y$embedded_words)) {
      means_of_embedded_words[x, ] = unlist(colMeans(y$embedded_words[[x]]))
    } 
    em_sim = lsa::cosine(t(means_of_embedded_words))
    diag(em_sim) = 0
    
    cosines_test = em_sim
    
    cosines_test <- cosines_test[,-1]
    values <- diag(cosines_test)
    values <- data.frame(values)
    values$id <- as.numeric(rownames(values))+1
    
    caption_data_full <- left_join(caption_data, values)
    caption_data_full$values <- replace_na(caption_data_full$values, 0)
    
    caption_data_full
    
  })   
  
  
  output$interestingTable <- renderDataTable({
    table <- CaptDataFull()
    table$id <- as.character(table$id)
    table %>% select(id, caption, values) %>% filter(values <= input$diff_value) %>% knitr::kable("html")
    
    
  })
  
  output$interestingPlot <- renderPlotly({
    
    plot_data <- CaptDataFull()
    diff_graph <- ggplot(plot_data, aes(id, values)) +
      geom_line(color = "#dd6818", size = 0.5) +
      geom_point(stat = "identity", size = 1, color = "#dd6818") +
      theme(axis.text.x = element_text(angle = 45, hjust = 1, colour = "white"), 
            axis.text.y = element_text(colour = "white"),
            panel.background = element_rect(fill = "#2b3e4f", colour = "#2b3e4f"),
            panel.grid.major = element_line(colour = "#2b3e4f"),
            panel.grid.minor = element_line(colour = "#2b3e4f"),
            panel.border = element_rect(colour = "black", fill=NA, size=0.5),
            plot.background = element_rect(fill = "#2b3e4f"),
            axis.title.x = element_text(colour = "white"),
            axis.title.y = element_text(colour = "white"))+
      scale_x_continuous(name = "Номер момента", breaks = seq(1, nrow(values)+1, 1)) +
      ylab("Похожесть момента на предыдущий")
    plotly::style(diff_graph, text = plot_data$caption)
    
  })
  
}

# Run the application 
shinyApp(ui = ui, server = server)
