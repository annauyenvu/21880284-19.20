'use strict';

let controller = {};
const models=require('../models');
const sequelize = require('sequelize');
const {Op} = require('sequelize');

controller.showHomepage = async (req, res) => {
    res.render('index'); 
}


controller.showPage = async (req, res) => {
    let keyword = req.query.keyword || '';
    let options = {
        attributes: ['id', 'title', 'summary', 'description', 'bigImagePath', 'smallImagePath'],
        where: {},
        include: [{
            model: models.Ingredient,
        }]
    }

    if (keyword.trim() !== '') {
        options.where = {
            [Op.or]: [
            { title: { [Op.iLike]: `%${keyword}%` } },
            { description: { [Op.iLike]: `%${keyword}%` } }
            ]
        };
    }

    let recipes = await models.Recipe.findAll(options);
    res.locals.recipes= recipes;  
    
    // Fetch ingredients for each recipe
    const recipesWithIngredients = await Promise.all(
        recipes.map(async (recipe) => {
            const ingredients = await recipe.getIngredients();
            return { ...recipe.toJSON(), ingredients };
        })
    );

    res.locals.recipes = recipesWithIngredients; // Update res.locals.recipes
    
    res.render('recipes');
}


controller.showDetails = async (req, res) => {
    let id = isNaN(req.params.id) ? 0 : parseInt(req.params.id);
    let recipe = await models.Recipe.findOne({
        attributes: ['id', 'title', 'description', 'bigImagePath', 'smallImagePath'],
        where: {id},
        include: [{
            model: models.Ingredient,
            attributes: ['id', 'quantity', 'title']
        }, {
            model:models.Direction,
            attributes: ['description', 'order'],
        }]
    });
    res.locals.recipe = recipe;
    console.log(recipe);
    res.render('featured');
}

module.exports = controller;