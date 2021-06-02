{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "KaggleLessons2.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyORobJtxWF6Ipb+/0hMq4rL",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/bruno353/Machine-Learning/blob/main/XGBRegressor_model.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "YciDCQEjqVUq"
      },
      "source": [
        "import pandas as pd"
      ],
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "bjmJUaoYvHz-"
      },
      "source": [
        "data = pd.read_csv(\"melb_data.csv\")"
      ],
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 343
        },
        "id": "hUkYHCM9vMNn",
        "outputId": "9415ee2a-c04c-4106-bf95-ba0fa563e443"
      },
      "source": [
        "data.head()"
      ],
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Suburb</th>\n",
              "      <th>Address</th>\n",
              "      <th>Rooms</th>\n",
              "      <th>Type</th>\n",
              "      <th>Price</th>\n",
              "      <th>Method</th>\n",
              "      <th>SellerG</th>\n",
              "      <th>Date</th>\n",
              "      <th>Distance</th>\n",
              "      <th>Postcode</th>\n",
              "      <th>Bedroom2</th>\n",
              "      <th>Bathroom</th>\n",
              "      <th>Car</th>\n",
              "      <th>Landsize</th>\n",
              "      <th>BuildingArea</th>\n",
              "      <th>YearBuilt</th>\n",
              "      <th>CouncilArea</th>\n",
              "      <th>Lattitude</th>\n",
              "      <th>Longtitude</th>\n",
              "      <th>Regionname</th>\n",
              "      <th>Propertycount</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>Abbotsford</td>\n",
              "      <td>85 Turner St</td>\n",
              "      <td>2</td>\n",
              "      <td>h</td>\n",
              "      <td>1480000.0</td>\n",
              "      <td>S</td>\n",
              "      <td>Biggin</td>\n",
              "      <td>3/12/2016</td>\n",
              "      <td>2.5</td>\n",
              "      <td>3067.0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>202.0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>Yarra</td>\n",
              "      <td>-37.7996</td>\n",
              "      <td>144.9984</td>\n",
              "      <td>Northern Metropolitan</td>\n",
              "      <td>4019.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>Abbotsford</td>\n",
              "      <td>25 Bloomburg St</td>\n",
              "      <td>2</td>\n",
              "      <td>h</td>\n",
              "      <td>1035000.0</td>\n",
              "      <td>S</td>\n",
              "      <td>Biggin</td>\n",
              "      <td>4/02/2016</td>\n",
              "      <td>2.5</td>\n",
              "      <td>3067.0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>156.0</td>\n",
              "      <td>79.0</td>\n",
              "      <td>1900.0</td>\n",
              "      <td>Yarra</td>\n",
              "      <td>-37.8079</td>\n",
              "      <td>144.9934</td>\n",
              "      <td>Northern Metropolitan</td>\n",
              "      <td>4019.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>Abbotsford</td>\n",
              "      <td>5 Charles St</td>\n",
              "      <td>3</td>\n",
              "      <td>h</td>\n",
              "      <td>1465000.0</td>\n",
              "      <td>SP</td>\n",
              "      <td>Biggin</td>\n",
              "      <td>4/03/2017</td>\n",
              "      <td>2.5</td>\n",
              "      <td>3067.0</td>\n",
              "      <td>3.0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>134.0</td>\n",
              "      <td>150.0</td>\n",
              "      <td>1900.0</td>\n",
              "      <td>Yarra</td>\n",
              "      <td>-37.8093</td>\n",
              "      <td>144.9944</td>\n",
              "      <td>Northern Metropolitan</td>\n",
              "      <td>4019.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>Abbotsford</td>\n",
              "      <td>40 Federation La</td>\n",
              "      <td>3</td>\n",
              "      <td>h</td>\n",
              "      <td>850000.0</td>\n",
              "      <td>PI</td>\n",
              "      <td>Biggin</td>\n",
              "      <td>4/03/2017</td>\n",
              "      <td>2.5</td>\n",
              "      <td>3067.0</td>\n",
              "      <td>3.0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>94.0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>Yarra</td>\n",
              "      <td>-37.7969</td>\n",
              "      <td>144.9969</td>\n",
              "      <td>Northern Metropolitan</td>\n",
              "      <td>4019.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>Abbotsford</td>\n",
              "      <td>55a Park St</td>\n",
              "      <td>4</td>\n",
              "      <td>h</td>\n",
              "      <td>1600000.0</td>\n",
              "      <td>VB</td>\n",
              "      <td>Nelson</td>\n",
              "      <td>4/06/2016</td>\n",
              "      <td>2.5</td>\n",
              "      <td>3067.0</td>\n",
              "      <td>3.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>120.0</td>\n",
              "      <td>142.0</td>\n",
              "      <td>2014.0</td>\n",
              "      <td>Yarra</td>\n",
              "      <td>-37.8072</td>\n",
              "      <td>144.9941</td>\n",
              "      <td>Northern Metropolitan</td>\n",
              "      <td>4019.0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "       Suburb           Address  ...             Regionname Propertycount\n",
              "0  Abbotsford      85 Turner St  ...  Northern Metropolitan        4019.0\n",
              "1  Abbotsford   25 Bloomburg St  ...  Northern Metropolitan        4019.0\n",
              "2  Abbotsford      5 Charles St  ...  Northern Metropolitan        4019.0\n",
              "3  Abbotsford  40 Federation La  ...  Northern Metropolitan        4019.0\n",
              "4  Abbotsford       55a Park St  ...  Northern Metropolitan        4019.0\n",
              "\n",
              "[5 rows x 21 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "G1HGwzuwvQKI",
        "outputId": "a79a45e7-39dc-4546-be55-d3433f360128"
      },
      "source": [
        "data.shape"
      ],
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(13580, 21)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 6
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 643
        },
        "id": "oL8lHZLuvSW1",
        "outputId": "0ca0f128-a3e1-4e86-b1f6-3115f0963a7a"
      },
      "source": [
        "data.dropna(axis = 0)"
      ],
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Suburb</th>\n",
              "      <th>Address</th>\n",
              "      <th>Rooms</th>\n",
              "      <th>Type</th>\n",
              "      <th>Price</th>\n",
              "      <th>Method</th>\n",
              "      <th>SellerG</th>\n",
              "      <th>Date</th>\n",
              "      <th>Distance</th>\n",
              "      <th>Postcode</th>\n",
              "      <th>Bedroom2</th>\n",
              "      <th>Bathroom</th>\n",
              "      <th>Car</th>\n",
              "      <th>Landsize</th>\n",
              "      <th>BuildingArea</th>\n",
              "      <th>YearBuilt</th>\n",
              "      <th>CouncilArea</th>\n",
              "      <th>Lattitude</th>\n",
              "      <th>Longtitude</th>\n",
              "      <th>Regionname</th>\n",
              "      <th>Propertycount</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>Abbotsford</td>\n",
              "      <td>25 Bloomburg St</td>\n",
              "      <td>2</td>\n",
              "      <td>h</td>\n",
              "      <td>1035000.0</td>\n",
              "      <td>S</td>\n",
              "      <td>Biggin</td>\n",
              "      <td>4/02/2016</td>\n",
              "      <td>2.5</td>\n",
              "      <td>3067.0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>156.0</td>\n",
              "      <td>79.00</td>\n",
              "      <td>1900.0</td>\n",
              "      <td>Yarra</td>\n",
              "      <td>-37.80790</td>\n",
              "      <td>144.99340</td>\n",
              "      <td>Northern Metropolitan</td>\n",
              "      <td>4019.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>Abbotsford</td>\n",
              "      <td>5 Charles St</td>\n",
              "      <td>3</td>\n",
              "      <td>h</td>\n",
              "      <td>1465000.0</td>\n",
              "      <td>SP</td>\n",
              "      <td>Biggin</td>\n",
              "      <td>4/03/2017</td>\n",
              "      <td>2.5</td>\n",
              "      <td>3067.0</td>\n",
              "      <td>3.0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>134.0</td>\n",
              "      <td>150.00</td>\n",
              "      <td>1900.0</td>\n",
              "      <td>Yarra</td>\n",
              "      <td>-37.80930</td>\n",
              "      <td>144.99440</td>\n",
              "      <td>Northern Metropolitan</td>\n",
              "      <td>4019.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>Abbotsford</td>\n",
              "      <td>55a Park St</td>\n",
              "      <td>4</td>\n",
              "      <td>h</td>\n",
              "      <td>1600000.0</td>\n",
              "      <td>VB</td>\n",
              "      <td>Nelson</td>\n",
              "      <td>4/06/2016</td>\n",
              "      <td>2.5</td>\n",
              "      <td>3067.0</td>\n",
              "      <td>3.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>120.0</td>\n",
              "      <td>142.00</td>\n",
              "      <td>2014.0</td>\n",
              "      <td>Yarra</td>\n",
              "      <td>-37.80720</td>\n",
              "      <td>144.99410</td>\n",
              "      <td>Northern Metropolitan</td>\n",
              "      <td>4019.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>Abbotsford</td>\n",
              "      <td>124 Yarra St</td>\n",
              "      <td>3</td>\n",
              "      <td>h</td>\n",
              "      <td>1876000.0</td>\n",
              "      <td>S</td>\n",
              "      <td>Nelson</td>\n",
              "      <td>7/05/2016</td>\n",
              "      <td>2.5</td>\n",
              "      <td>3067.0</td>\n",
              "      <td>4.0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>245.0</td>\n",
              "      <td>210.00</td>\n",
              "      <td>1910.0</td>\n",
              "      <td>Yarra</td>\n",
              "      <td>-37.80240</td>\n",
              "      <td>144.99930</td>\n",
              "      <td>Northern Metropolitan</td>\n",
              "      <td>4019.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>Abbotsford</td>\n",
              "      <td>98 Charles St</td>\n",
              "      <td>2</td>\n",
              "      <td>h</td>\n",
              "      <td>1636000.0</td>\n",
              "      <td>S</td>\n",
              "      <td>Nelson</td>\n",
              "      <td>8/10/2016</td>\n",
              "      <td>2.5</td>\n",
              "      <td>3067.0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>256.0</td>\n",
              "      <td>107.00</td>\n",
              "      <td>1890.0</td>\n",
              "      <td>Yarra</td>\n",
              "      <td>-37.80600</td>\n",
              "      <td>144.99540</td>\n",
              "      <td>Northern Metropolitan</td>\n",
              "      <td>4019.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>12205</th>\n",
              "      <td>Whittlesea</td>\n",
              "      <td>30 Sherwin St</td>\n",
              "      <td>3</td>\n",
              "      <td>h</td>\n",
              "      <td>601000.0</td>\n",
              "      <td>S</td>\n",
              "      <td>Ray</td>\n",
              "      <td>29/07/2017</td>\n",
              "      <td>35.5</td>\n",
              "      <td>3757.0</td>\n",
              "      <td>3.0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>972.0</td>\n",
              "      <td>149.00</td>\n",
              "      <td>1996.0</td>\n",
              "      <td>Whittlesea</td>\n",
              "      <td>-37.51232</td>\n",
              "      <td>145.13282</td>\n",
              "      <td>Northern Victoria</td>\n",
              "      <td>2170.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>12206</th>\n",
              "      <td>Williamstown</td>\n",
              "      <td>75 Cecil St</td>\n",
              "      <td>3</td>\n",
              "      <td>h</td>\n",
              "      <td>1050000.0</td>\n",
              "      <td>VB</td>\n",
              "      <td>Williams</td>\n",
              "      <td>29/07/2017</td>\n",
              "      <td>6.8</td>\n",
              "      <td>3016.0</td>\n",
              "      <td>3.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>179.0</td>\n",
              "      <td>115.00</td>\n",
              "      <td>1890.0</td>\n",
              "      <td>Hobsons Bay</td>\n",
              "      <td>-37.86558</td>\n",
              "      <td>144.90474</td>\n",
              "      <td>Western Metropolitan</td>\n",
              "      <td>6380.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>12207</th>\n",
              "      <td>Williamstown</td>\n",
              "      <td>2/29 Dover Rd</td>\n",
              "      <td>1</td>\n",
              "      <td>u</td>\n",
              "      <td>385000.0</td>\n",
              "      <td>SP</td>\n",
              "      <td>Williams</td>\n",
              "      <td>29/07/2017</td>\n",
              "      <td>6.8</td>\n",
              "      <td>3016.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>35.64</td>\n",
              "      <td>1967.0</td>\n",
              "      <td>Hobsons Bay</td>\n",
              "      <td>-37.85588</td>\n",
              "      <td>144.89936</td>\n",
              "      <td>Western Metropolitan</td>\n",
              "      <td>6380.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>12209</th>\n",
              "      <td>Windsor</td>\n",
              "      <td>201/152 Peel St</td>\n",
              "      <td>2</td>\n",
              "      <td>u</td>\n",
              "      <td>560000.0</td>\n",
              "      <td>PI</td>\n",
              "      <td>hockingstuart</td>\n",
              "      <td>29/07/2017</td>\n",
              "      <td>4.6</td>\n",
              "      <td>3181.0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>61.60</td>\n",
              "      <td>2012.0</td>\n",
              "      <td>Stonnington</td>\n",
              "      <td>-37.85581</td>\n",
              "      <td>144.99025</td>\n",
              "      <td>Southern Metropolitan</td>\n",
              "      <td>4380.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>12212</th>\n",
              "      <td>Yarraville</td>\n",
              "      <td>54 Pentland Pde</td>\n",
              "      <td>6</td>\n",
              "      <td>h</td>\n",
              "      <td>2450000.0</td>\n",
              "      <td>VB</td>\n",
              "      <td>Village</td>\n",
              "      <td>29/07/2017</td>\n",
              "      <td>6.3</td>\n",
              "      <td>3013.0</td>\n",
              "      <td>6.0</td>\n",
              "      <td>3.0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>1087.0</td>\n",
              "      <td>388.50</td>\n",
              "      <td>1920.0</td>\n",
              "      <td>Maribyrnong</td>\n",
              "      <td>-37.81038</td>\n",
              "      <td>144.89389</td>\n",
              "      <td>Western Metropolitan</td>\n",
              "      <td>6543.0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>6196 rows × 21 columns</p>\n",
              "</div>"
            ],
            "text/plain": [
              "             Suburb          Address  ...             Regionname Propertycount\n",
              "1        Abbotsford  25 Bloomburg St  ...  Northern Metropolitan        4019.0\n",
              "2        Abbotsford     5 Charles St  ...  Northern Metropolitan        4019.0\n",
              "4        Abbotsford      55a Park St  ...  Northern Metropolitan        4019.0\n",
              "6        Abbotsford     124 Yarra St  ...  Northern Metropolitan        4019.0\n",
              "7        Abbotsford    98 Charles St  ...  Northern Metropolitan        4019.0\n",
              "...             ...              ...  ...                    ...           ...\n",
              "12205    Whittlesea    30 Sherwin St  ...      Northern Victoria        2170.0\n",
              "12206  Williamstown      75 Cecil St  ...   Western Metropolitan        6380.0\n",
              "12207  Williamstown    2/29 Dover Rd  ...   Western Metropolitan        6380.0\n",
              "12209       Windsor  201/152 Peel St  ...  Southern Metropolitan        4380.0\n",
              "12212    Yarraville  54 Pentland Pde  ...   Western Metropolitan        6543.0\n",
              "\n",
              "[6196 rows x 21 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ho7Lthw0p-Xt"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "SWmSzzHvnokW"
      },
      "source": [
        "X = data[['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt', 'Bathroom', 'Bedroom2', 'Lattitude', 'Longtitude']]"
      ],
      "execution_count": 25,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "R7ZMMXxloWT_"
      },
      "source": [
        "Y = data.Price"
      ],
      "execution_count": 26,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "7zpqtF9EoeGB"
      },
      "source": [
        "from sklearn.model_selection import train_test_split"
      ],
      "execution_count": 27,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "GUhLJvC4oiHw"
      },
      "source": [
        "X_train, X_test, Y_train, Y_test = train_test_split(X, Y)"
      ],
      "execution_count": 28,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ysjDeTb2o0Jm"
      },
      "source": [
        "from xgboost import XGBRegressor"
      ],
      "execution_count": 29,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "wj1KRR2qo2tW"
      },
      "source": [
        "model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)"
      ],
      "execution_count": 30,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "dIa4RZZ9o8DV",
        "outputId": "4f0fdfb0-63b0-499b-a42d-522f30922a7b"
      },
      "source": [
        "model.fit(X_train, Y_train, \n",
        "             early_stopping_rounds=5, \n",
        "             eval_set=[(X_test, Y_test)], \n",
        "             verbose=False)"
      ],
      "execution_count": 31,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[16:59:24] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,\n",
              "             colsample_bynode=1, colsample_bytree=1, gamma=0,\n",
              "             importance_type='gain', learning_rate=0.05, max_delta_step=0,\n",
              "             max_depth=3, min_child_weight=1, missing=None, n_estimators=1000,\n",
              "             n_jobs=4, nthread=None, objective='reg:linear', random_state=0,\n",
              "             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,\n",
              "             silent=None, subsample=1, verbosity=1)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 31
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "y4dFOzyUpCpT"
      },
      "source": [
        "predicao = model.predict(X_test)"
      ],
      "execution_count": 32,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Hp72LEmIpf1Y"
      },
      "source": [
        "from sklearn.metrics import mean_absolute_error"
      ],
      "execution_count": 33,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4SLe1SIPpmT-",
        "outputId": "8a0c3f49-9f5b-4513-dcff-01bfd19232b8"
      },
      "source": [
        "mean_absolute_error(predicao, Y_test)"
      ],
      "execution_count": 34,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "267146.28970913106"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 34
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "oa60wCM8p1h8"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}