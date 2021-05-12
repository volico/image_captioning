library(shiny)
library(httr)
library(rjson)
library(jsonlite)
library(dplyr)
library(stringr)
library(tidytext)
library(stopwords)
library(tidyr)
library(ggplot2)
library(plotly)
library(shinythemes)
library(DT)
library(lsa)
library(lubridate)
library(scales)
library(Hmisc)

options(shiny.maxRequestSize=1024*1024^2)
def_video="/Users/simon/Desktop/for_shiny/Pig.mp4"

# Define UI for application that draws a histogram
ui <- tagList(
  fluidPage(
    theme=shinytheme("superhero"),
    # Application title
    titlePanel("Image Captioning"),
    
    # Sidebar with a slider input for number of bins
    sidebarLayout(
      sidebarPanel(
        fileInput("file", label = h3("Upload your file here")),
        tags$style("
                   .btn-file {  
                   background-color:#FF8C00; 
                   border-color: #FF8C00; 
                   }
                   
                   .progress-bar {
                   background-color: #FF8C00;
                   }
                   
                   "),
        sliderInput("diff_value", "Establish the coefficient of similarity", 0, 1, 0),
        actionButton("action","Start the analysis", icon("angle-double-right"), 
                     style="color: #fff; background-color: #337ab7; border-color: #2e6da4"),
        hr(),
        p('Hi there! Welcome to "Image Captioning" application. Here you can check your video 
          recordings for strange moments (when something irregular happens). You do not need to 
          watch tones of video by yourself anymore. In the graph on the right you will see 
          the comparisons of each 10th second`s frame with the previous ones. If a dot is too high,
          probably something extraordinary occured in a video. Place the cursor in a dot and
          get an approximate description of the frame. The table with all decriptions 
          is above the graph.'),
         p('You can see the analysis of the "Example video" (here we will put the link). If you want do check your own video press the "Browse" button.
         Be patient - loading may take some time. Please, wait until the panel answer "Analysis is done" below the graph.'),
        p('Thank you for using "Image Captioning!"'),
        
        actionButton("action2","About us", icon("address-card"), 
                     style="color: #fff; background-color: #337ab7; border-color: #2e6da4"),
        conditionalPanel(condition="input.action2",p('Head manager: Sam Zabrodin'), 
                                                   p('  Executive producer: Sam Zabrodin'),
                                                   p('  The author of the idea: Sam Zabrodin'),
                                                   p('  Financial director: Sam Zabrodin'),
                                                   p('  Some other dudes, who sometimes 
                                                     appeared on breefings and
                                                     done almost 3% of all the job: 
                                                     Ilya, Varya, Ruslan'))
        ),
      
      # Show a plot of the generated distribution
      
     mainPanel(
        tags$style(HTML("
                        .dataTables_wrapper .dataTables_length, .dataTables_wrapper .dataTables_filter, .dataTables_wrapper .dataTables_info, .dataTables_wrapper .dataTables_processing, .dataTables_wrapper .dataTables_paginate {
                        color: #ffffff;
                        }
                        
                        thead {
                        color: #ffffff;
                        }
                        
                        tbody {
                        color: #000000;
                        }
                        
                        ")),
        tags$head(tags$style(type="text/css", "
             #loadmessage {
                             position: fixed;
                             top: 0px;
                             left: 500px;
                             width: 50%;
                             padding: 5px 0px 5px 0px;
                             text-align: center;
                             font-weight: bold;
                             font-size: 100%;
                             color: #FFFFFF;
                             z-index: 105;
                             }
                             ")),
        conditionalPanel(condition="$('html').hasClass('shiny-busy')",
                         tags$div("Loading...",id="loadmessage")),
        dataTableOutput("interestingTable"),
        
        plotlyOutput("interestingPlot"),
        textOutput("done")
        )    

        )
      ))

# Define server logic required to draw a histogram
server <- function(input, output) {
  
  List <- reactive ({
    if (input$action){
      path=input$file$datapath
      }
    if (!input$action){path="/Users/simon/Desktop/for_shiny/Pig.mp4"}
    x = POST("http://94.19.209.176:8888/get_video_captions", body = (list(video = upload_file(path))))
    lst = jsonlite::fromJSON(rawToChar(x$content)) #input$file$datapath
    lst
  })
  
  CaptData <- reactive({
    z <- List()
    caption <- z$captions
    data <- data.frame(caption, stringsAsFactors = FALSE)
    data$id <- as.numeric(rownames(data))
    data$caption <- capitalize(data$caption)
    data
    
  })
  
  CaptDataFull <- reactive({
    caption_data_full <- CaptData()
    y <- List()
    
    caption_data_full$distance <- y$distance_from_previous
    
    caption_data_full$values <- scales::rescale(caption_data_full$distance)

    caption_data_full$time <- (caption_data_full$id-1)*10
    
    caption_data_full$time <- as.duration(caption_data_full$time)
    
    caption_data_full$time <- seconds_to_period(caption_data_full$time)
    
    caption_data_full$time <- sprintf('%02d:%02d', minute(caption_data_full$time), second(caption_data_full$time))
    
    caption_data_full
    
  })   
  
  
  output$interestingTable <- renderDataTable({
    table <- CaptDataFull()
    table$id <- as.character(table$id)
    table %>% select(time, caption, distance, values) %>% filter(values >= input$diff_value) %>% 
      datatable(colnames = c("Timing", "Caption", "Euclidean distance", "Scaled distance"), 
                rownames = FALSE, class = 'cell-border hover', 
                caption = htmltools::tags$caption("The most interesting moments", style = 'caption-side: top; text-align: left; color:white; font-size: 20px;'))
  })
  
  output$interestingPlot <- renderPlotly({
    
    plot_data <- CaptDataFull()
    diff_graph <- ggplot(plot_data, aes(time, values, group = 1)) +
      geom_line(color = "#dd6818", size = 0.5) +
      geom_point(stat = "identity", size = 1, color = "#dd6818") +
      theme(axis.text.x = element_text(angle = 45, hjust = 1, colour = "white"), 
            axis.text.y = element_text(colour = "white"),
            panel.background = element_rect(fill = "#2b3e4f", colour = "#2b3e4f"),
            panel.grid.major = element_line(colour = "#2b3e4f"),
            panel.grid.minor = element_line(colour = "#2b3e4f"),
            panel.border = element_rect(colour = "black", fill=NA, size=0.5),
            plot.background = element_rect(fill = "#2b3e4f"),
            plot.title = element_text(colour = "white"),
            axis.title.x = element_text(colour = "white"),
            axis.title.y = element_text(colour = "white")) +
      xlab("Timing") +
      ylab("Similarity coefficient") +
      ggtitle("Plot of similarity to the previous moments")
    plotly::style(diff_graph, text = plot_data$caption)
    
  })
  
  output$done=renderText('Analysis is done')
  
  
  
}

# Run the application 
shinyApp(ui = ui, server = server)

