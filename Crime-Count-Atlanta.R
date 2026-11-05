# Installa il pacchetto se non lo hai già
if (!require("arrow")) install.packages("arrow")
if (!require("pscl")) install.packages("pscl")
if (!require("fitdistrplus")) install.packages("fitdistrplus")
library(arrow)
library(ggplot2)
library(gridExtra)
library(dplyr)
library(pscl)
library(tidyr)

dat <- read_parquet("npu_dense_panel.parquet")
str(dat)

# Modifichiamo il dataset lasciando solo le variabili più rilevanti e fattorizzando le variabili categoriali

dat_clean <- dat %>%
  dplyr::select(-grid_density_7d, -location_type_count,
                -day_of_week, -is_cold) %>%
  mutate(across(c(starts_with(c("is_", "near_")), npu), as.factor))
glimpse(dat_clean)

dat_clean <- dat_clean %>%
  arrange(hour_ts)

dat_an <- dat_clean %>%
  dplyr::select(crime_count, npu,npu_crime_avg_30d,temp_f, near_gsu, near_ga_tech, near_emory, near_clark,
                near_spelman, near_morehouse, near_morehouse_med, near_atlanta_metro, near_atlanta_tech, near_scad, near_spelman, near_john_marshall)

summary(dat_an)
#
p1<- dat_clean %>%
  mutate(is_zero = ifelse(crime_count == 0, "Zeri", "Valori > 0")) %>%
  ggplot(aes(x = is_zero)) +
  geom_bar(fill = "steelblue") +
  labs(title = "Distribuzione Zeri vs Non-Zeri", x = "crime_count", y = "Conteggio")

# 2. Visualizza la distribuzione della "coda" (escludendo gli zeri)
p2<- dat_clean %>%
  filter(crime_count > 0) %>%
  ggplot(aes(x = crime_count)) +
  geom_bar(fill = "firebrick", color = "black") +
  labs(title = "Distribuzione dei soli valori positivi di crime_count", y= "Conteggio")

grid.arrange(p1, p2, ncol = 2)

sum(dat_clean$crime_count==0)/nrow(dat_clean) #90.65% DI ORE SENZA CRIMINI IN 4 ANNI
sum(dat_clean$crime_count>0)/nrow(dat_clean)  #9.35% DI ORE CON CRIMINI IN 4 ANNI PARI A 100'497.

dat_pos <- dat_an[dat_clean$crime_count>0,]

# 3. Stima dei modelli
# Poisson
fit.po <- glm(crime_count~.-hour_ts- is_raining - is_hot - is_daylight - is_holiday - month -
                year,
              family=poisson(log),
              data=dat_clean)
summary(fit.po)

# Zero-Inflated Poisson
fit.zip <- zeroinfl(crime_count ~ near_gsu + near_ga_tech + npu_crime_avg_30d +
                      near_emory + near_clark + near_spelman + near_morehouse +
                      near_atlanta_metro + near_scad + near_john_marshall|
                      1 + npu,
                    data=dat_clean)
summary(fit.zip)

res_pearson_zip <- residuals(fit.zip, type = "pearson")

dispersion_zip <- sum(res_pearson_zip^2) / df.residual(fit.zip)
dispersion_zip



set.seed(917031)
n_train <- floor(0.80 * nrow(dat_clean))
train_set <- dat_clean[1:n_train, ]
test_set <- dat_clean[(n_train + 1):nrow(dat_clean), ]

x_test <- test_set[,-3]
y_test <- test_set[,3]

# Stima dei modelli sul training set
fit.po <- glm(crime_count~.-hour_ts- is_raining - is_hot - is_daylight - is_holiday - month -
                year,
              family=poisson(log),
              data=train_set)
summary(fit.po)

fit.zip <- zeroinfl(crime_count ~ npu + near_gsu + near_ga_tech +
                      near_emory + near_clark + near_spelman + near_morehouse +
                      near_atlanta_metro + near_scad + near_john_marshall|
                      1 + npu_crime_avg_30d,
                    data=train_set)
summary(fit.zip)

fit.hurdle<- hurdle(crime_count ~ npu + near_gsu + near_ga_tech +
                      near_emory + near_clark + near_spelman + near_morehouse +
                      near_atlanta_metro + near_scad + near_john_marshall|
                      1 + npu_crime_avg_30d,
                    data=train_set, dist="poisson",
                    zero.dist="binomial",
                    link="logit")
summary(fit.hurdle)

# Previsioni e calcolo delle metriche RMSE e MAE
pred_po <- predict(fit.po,   newdata = x_test, type = "response")
pred_zip <- predict(fit.zip, newdata = x_test, type = "response")
pred_hurdle <- predict(fit.hurdle, newdata = x_test, type = "response")

pred_vs_real_po <- data.frame(y_test, pred_po)
pred_vs_real_zip <- data.frame(y_test, pred_zip)
pred_vs_real_hurdle <- data.frame(y_test, pred_hurdle)

par(mfrow = c(1,2))
barplot(pred_vs_real_po[,2])
barplot(pred_vs_real_po[,1])

par(mfrow = c(1,2))
barplot(pred_vs_real_zip[,2], xlab = "Previsioni ZIP", ylim = c(0,5))
barplot(pred_vs_real_zip[,1], xlab = "Valori reali")

pred_rounded <- round(pred_vs_real_zip$pred_zip)

par(mfrow = c(1,2))
barplot(pred_vs_real_hurdle[,2], xlab = "Previsioni Hurdle")
barplot(pred_vs_real_hurdle[,1], xlab = "Valori reali")

pred_rounded_hurdle <- round(pred_vs_real_hurdle$pred_hurdle)

tabella_performance <- data.frame(
  RMSE = c(
    RMSE(pred_po, y_test$crime_count),
    RMSE(pred_zip, y_test$crime_count),
    RMSE(pred_hurdle, y_test$crime_count)
  ),
  MAE = c(
    MAE(pred_po, y_test$crime_count),
    MAE(pred_zip, y_test$crime_count),
    MAE(pred_hurdle,y_test$crime_count)
  )
)

# Assegniamo i nomi alle righe (i modelli)
row.names(tabella_performance) <- c("Poisson", "ZIP", "Hurdle")

# Visualizziamo la tabella
print(tabella_performance[order(tabella_performance$RMSE),])