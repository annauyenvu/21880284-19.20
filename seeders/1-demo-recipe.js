'use strict';

module.exports = {
    up: (queryInterface, Sequelize) => {
        var recipes = [{
            title: 'Cake',
            smallImagePath: '/images/cake.jpg',
            bigImagePath: '/images/video1.jpg',
            summary: 'Lorem ipsum dolor sit amet consectetur adipisicing elit. Laudantium dolores aliquid facere numquam quibusdam atque excepturi quasi delectus placeat eos!',
            description: 'Lorem ipsum dolor sit amet consectetur adipisicing elit. Blanditiis fuga qui facere. Architecto facilis ut perferendis provident a vel nisi amet, maxime aspernatur eveniet ex cupiditate officiis, quasi, sequi sed.'
        },
        {
            title: 'Fried chicken',
            smallImagePath: '/images/fried-chicken.jpg',
            bigImagePath: '/images/turkey.jpg',
            summary: 'Lorem, ipsum dolor sit amet consectetur adipisicing elit. Provident ipsam recusandae, sint tempore necessitatibus veritatis autem harum quibusdam tenetur dignissimos?',
            description: 'Lorem ipsum dolor sit amet consectetur adipisicing elit. Qui quam cum dignissimos mollitia, alias ab accusamus molestias. Ea at quasi est tenetur veritatis laboriosam cumque enim, excepturi obcaecati perferendis repellendus!'
        }, {
            title: 'Pastries',
            smallImagePath: '/images/pastries.jpg',
            bigImagePath: '/images/pork-barbeque.jpg',
            summary: 'Lorem ipsum dolor sit amet consectetur adipisicing elit. Esse architecto vel, molestiae ab minima a dolores repudiandae. Illum, nisi? Officiis!',
            description: 'Lorem ipsum dolor sit amet consectetur adipisicing elit. Excepturi, impedit inventore distinctio autem doloribus beatae recusandae, iste deserunt soluta dolorem, quibusdam fugiat aliquid. Sunt culpa recusandae earum ipsa similique quibusdam.'
        }, {
            title: 'Rice Topping',
            smallImagePath: '/images/rice-topping.jpg',
            bigImagePath: '/images/chicken-mushroom.jpg',
            summary: 'Lorem, ipsum dolor sit amet consectetur adipisicing elit. Necessitatibus soluta, iste minus debitis porro cupiditate voluptatibus maiores odio totam consequuntur?',
            description: 'Lorem ipsum dolor sit, amet consectetur adipisicing elit. Commodi, eum sit. Dolorum, quam! Accusamus voluptate fugit minus, eos, saepe atque aspernatur quidem dicta quos aperiam sint repellendus optio ipsa quae.'
        }, {
            title: 'Strawberry Juice',
            smallImagePath: '/images/strawberry-juice.jpg',
            bigImagePath: '/images/video2.jpg',
            summary: 'Lorem ipsum dolor sit amet, consectetur adipisicing elit. Harum laboriosam aliquid quae assumenda architecto eius possimus minima veritatis nesciunt obcaecati.',
            description: 'Lorem ipsum dolor sit amet consectetur adipisicing elit. Necessitatibus, optio nihil expedita fugit placeat praesentium accusamus veniam fuga nesciunt quis, quisquam blanditiis exercitationem! Nisi laudantium sit, cupiditate veritatis quaerat nesciunt?'
        }];

        recipes.forEach((item, index) => {
            item.createdAt = Sequelize.literal('NOW()');
            item.updatedAt = Sequelize.literal('NOW()');
        });
        return queryInterface.bulkInsert('Recipes', recipes, {});

    },

    down: (queryInterface, Sequelize) => {
        return queryInterface.bulkDelete('Recipes', null, {});
    }
};