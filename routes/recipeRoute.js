
'use strict';
const express = require('express');
const router = express.Router();
const controller = require('../controllers/recipeController');

router.get('/', controller.showHomepage);
router.get('/recipes', controller.showPage);
router.get('/recipes/:id', controller.showDetails);

module.exports = router; 