'use strict';

module.exports = {
    up: (queryInterface, Sequelize) => {
        var directions = [];
        var descriptions = [
           ' Lorem ipsum dolor sit amet.',
            'Lorem ipsum dolor sit amet consectetur.',
            'Lorem ipsum dolor sit, amet consectetur adipisicing.',
            'Lorem ipsum dolor sit amet consectetur adipisicing elit.',    
            'Lorem ipsum dolor, sit amet consectetur adipisicing elit. Pariatur?',
            'Lorem ipsum dolor sit, amet consectetur adipisicing elit. Magni, corporis.'
        ];
        var i, j, k;
        for (i = 1; i < 6; i++) {
            var count = Math.floor(Math.random() * 6) + 4;
            for (j = count-1; j >=0 ; j--) {
                k = Math.floor(Math.random() * 6);
                directions.push({
                    RecipeId: i,
                    order: j,
                    description: descriptions[k],
                    createdAt: Sequelize.literal('NOW()'),
                    updatedAt: Sequelize.literal('NOW()')
                });
            }
        }
        return queryInterface.bulkInsert('Directions', directions, {});
    },

    down: (queryInterface, Sequelize) => {
        return queryInterface.bulkDelete('Directions', null, {});
    }
};