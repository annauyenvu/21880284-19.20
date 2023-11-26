'use strict';
const {
  Model
} = require('sequelize');
module.exports = (sequelize, DataTypes) => {
  class Recipe extends Model {
    /**
     * Helper method for defining associations.
     * This method is not a part of Sequelize lifecycle.
     * The `models/index` file will call this method automatically.
     */
    static associate(models) {
      // define association here
      Recipe.hasMany(models.Ingredient);
        Recipe.hasMany(models.Direction);
    }
  }
  Recipe.init({
    title: DataTypes.STRING,
    smallImagePath: DataTypes.STRING,
    bigImagePath: DataTypes.STRING,
    summary: DataTypes.STRING,
    description: DataTypes.TEXT
  }, {
    sequelize,
    modelName: 'Recipe',
  });
  return Recipe;
};