// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import App from './App'
import router from './router'
import store from './store'

import io from 'socket.io-client'
import VueSocketIO from 'vue-socket.io'
import VueMaterial from 'vue-material'
import 'vue-material/dist/vue-material.min.css'
import 'vue-material/dist/theme/default.css'

Vue.config.productionTip = false

export const SocketInstance = io(window.location.protocol + '//' + window.location.hostname + ':5000', { secure: true })
// export const SocketInstance = io()

Vue.use(VueMaterial)

Vue.use(new VueSocketIO({
  debug: true,
  connection: SocketInstance
}))

/* eslint-disable no-new */
new Vue({
  router,
  store,
  render: h => h(App),
  components: { App },
  template: '<App/>'
}).$mount('#app')
