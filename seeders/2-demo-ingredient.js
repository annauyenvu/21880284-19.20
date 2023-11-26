'use strict';

function getRandomInt(max) {
    return Math.floor(Math.random() * Math.floor(max));
}

module.exports = {
    up: (queryInterface, Sequelize) => {
        var ingredients = [];
        var quantities = ['10 grms', '12 pcs', '3/4 cup', '6 slices', '1 liter', '1/4 cup'];
        var titles = [
            'Possimus sit amet.',
            ' Nihil amet consectetur.',
            'Praesentium consectetur adipisicing.',
            'Repudiandae ipsum dolor.',
            'Deleniti dolor sit.',
            'Quos Lorem ipsum.'
        ];
        
        var count = Math.floor(Math.random() * 6) + 3;
        var i, j, k;
        for (i = 1; i < 6; i++) {
            for (j = 0; j < count; j++) {
                k = Math.floor(Math.random() * 6);
                ingredients.push({
                    RecipeId: i,
                    quantity: quantities[k],
                    title: titles[k],
                    createdAt: Sequelize.literal('NOW()'),
                    updatedAt: Sequelize.literal('NOW()')
                });
            }
        }
        return queryInterface.bulkInsert('Ingredients', ingredients, {});
    },

    down: (queryInterface, Sequelize) => {
        return queryInterface.bulkDelete('Ingredients', null, {});

    }
};