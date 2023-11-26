'use strict';

const express = require("express");
const app = express();
const port = 5000; 
const handlebars = require('express-handlebars');
const {connect} = require('./routes/recipeRoute');

// let models = require('./models');
// models.sequelize.sync().then(() => {
//     console.log('table created!');
// })

//cau hinh public static folder
app.use(express.static(__dirname+"/public"));

//cau hinh sd express handlebars
app.engine('hbs', handlebars.engine({//dn engine hbs sd engine của handlebars
    extname: 'hbs', //cac file template sau nay se co duoi file hbs
    defaultLayout: "layout", //file layout chinh duoc dat ten la layout
    partialsDir: __dirname+"/views/partials", //cac layout con nam o day  
    layoutsDir: __dirname+"/views/layouts", //file layout chinh nam o day
    runtimeOptions: {
        allowProtoPropertiesByDefault: true,
    },
    helpers: {
        isEven: function(index) {
            return index % 2 === 0;
        },
        compare: function (v1, operator, v2, options) {
            switch (operator) {
                case '==':
                    return (v1 == v2) ? options.fn(this) : options.inverse(this);
                case '===':
                    return (v1 === v2) ? options.fn(this) : options.inverse(this);
                case '!=':
                    return (v1 != v2) ? options.fn(this) : options.inverse(this);
                case '!==':
                    return (v1 !== v2) ? options.fn(this) : options.inverse(this);
                case '<':
                    return (v1 < v2) ? options.fn(this) : options.inverse(this);
                case '<=':
                    return (v1 <= v2) ? options.fn(this) : options.inverse(this);
                case '>':
                    return (v1 > v2) ? options.fn(this) : options.inverse(this);
                case '>=':
                    return (v1 >= v2) ? options.fn(this) : options.inverse(this);
                default:
                    return options.inverse(this);
            }
        }
    }
}))
app.set("view engine", "hbs");
//cau hinh doc dl post tu req.body
app.use(express.json());
app.use(express.urlencoded({
    extended: false,
}));

//routes
app.use("/", require('./routes/recipeRoute'));

app.use((req, res, next) => {
    res.status(404).render('error', {message: 'File not Found'});
});
app.use((error, req, res, next) => {
    console.error(error);
    res.status(500).render('error', {message: 'Internal Server Error'});
});

//Khoi dong web server 
app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
})