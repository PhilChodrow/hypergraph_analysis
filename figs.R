
# install.packages(c('tidyverse', 'knitr'))

# create directory to hold figs if one doesn't already exist
dir.create(file.path('figs'))

library(tidyverse)

# Triangles: corresponds to Table 1 in manuscript

datasets <- c('email-Enron')

df <- datasets %>% 
    map(~paste0('throughput/', ., '/clustering.csv')) %>% 
    map(read_csv) %>% 
    reduce(rbind) %>% 
    filter(label %in% c('empirical' ) | i > 100) 

df %>% 
    filter(!(label == 'empirical' & graph == 'projected')) %>% 
    group_by(data, label, graph) %>% 
    summarise(s = sd(C), C = mean(C)) %>% 
    mutate(disp = paste0(sprintf("%.5f",C), ' (',sprintf("%.5f",s), ')')) %>% 
    select(-s, -C) %>% 
    spread(key = data, value = disp) %>% 
    mutate(graph = ifelse(label == 'empirical', '', graph)) %>% 
    arrange(graph) %>% 
    knitr::kable()




# Assortativity: corresponds to Fig. 2 in manuscript

df <- datasets %>%
    map(~paste0('throughput/', ., '/assortativity.csv')) %>% 
    map(read_csv) %>% 
    reduce(rbind) 


empirical_df <- df %>% 
    filter(grepl(pattern = 'empirical', label)) %>% 
    filter(cor == 'spearman', 
           label == 'empirical', 
           choice %in% c('uniform', NA)) %>% 
    mutate(graph = stringr::str_to_title(graph))

plot_df <- df %>% 
    filter(cor == 'spearman') %>% 
    mutate(choice = ifelse(is.na(choice), 'uniform_graph', choice)) %>%
    mutate(choice = stringr::str_to_title(choice),
           choice = stringr::str_replace(choice, '_', '-'),
           choice = ifelse(choice == 'Uniform-graph', 'Projected', choice),
           choice = recode(choice, `Top-bottom` = "Top-Bottom"))

plot_df <- plot_df %>% 
    mutate(choice = factor(choice, levels = c(
        'Projected', 
        'Uniform', 
        'Top-2', 
        'Top-Bottom'
    ))) 

empirical_df <- plot_df %>%
    filter(label == 'empirical')

plot_df <- plot_df %>% 
    filter(label != 'empirical')

p <- plot_df %>% 
    ggplot() + 
    aes(x = coef, color = stringr::str_to_title(label)) + 
    geom_density(fill = 'black', alpha = .1, trim = T) + 
    facet_wrap(~choice, nrow = 1) + 
    geom_vline(aes(xintercept = coef), data = empirical_df, show.legend = F, linetype = 'dashed') + 
    xlab(expression(paste("Spearman's ", rho))) + 
    guides(color = guide_legend(title = 'Model space'))
p
ggsave('figs/assortativity.png', width = 7, height = 3)


# Intersection Profiles: corresponds to Fig. 3b in manuscript

data <- 'email-Enron'
data_path <- paste0('throughput/', data, '/intersection.csv')
df <- read_csv(data_path)

df <- df %>% 
    filter(j <= 10)

# symmetrize

append_df <- df %>% 
    mutate(K1 = k_1, K2 = k_2) %>% 
    mutate(k_2 = K1, k_1 = K2) %>% 
    select(-K1, -K2)

df <- df %>% 
    rbind(append_df)


df <- df %>% 
    filter(j <= k_1, j <= k_2)

empirical_df <- df %>% 
    filter(label == 'empirical') %>% 
    group_by(data, j) %>% 
    summarise(n = sum(n)) %>% 
    mutate(p = n / sum(n))

sim_df <- df %>% 
    filter(label != 'empirical') %>% 
    mutate(n = as.numeric(n)) %>% 
    group_by(k_1, k_2, j, data, label) %>%  
    summarise( sd = sd(n, na.rm = T), n = sum(n)) %>% 
    group_by(data, label, j) %>% 
    summarise(n = sum(n)) %>% 
    group_by(data, label) %>% 
    mutate(p = n/sum(n)) 

y_breaks <- 10^(0:round(min(log10(sim_df$p), na.rm = T)))

q <- sim_df %>% 
    ggplot() + 
    aes(x = j, y = p) + 
    geom_line(aes(color = label)) + 
    geom_point(data = empirical_df, color = 'grey20') + 
    scale_y_continuous(trans = 'log10', breaks = y_breaks) + 
    scale_x_continuous(breaks = 0:max(df$j)) + 
    theme_bw() + 
    scale_color_manual(values = c('grey20', 'navy', 'orange')) + 
    guides(linetype = FALSE, 
           color = guide_legend(title = element_blank())) + 
    theme(
        legend.background = element_rect(fill = scales::alpha('white', 0)), 
        legend.justification = c(1,1),
        legend.position = c(1,1),
        panel.grid.minor = element_blank()) + 
    ylab(expression(italic(r(j~~'|'~~H)))) + 
    xlab(expression(italic(j)))

ggsave('figs/intersection.png', width = 5, height = 3)