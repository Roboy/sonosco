// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import App from './App'
import router from './router'
import store from './store'

import io from 'socket.io-client'
import VueSocketIO from 'vue-socket.io'
import VueMaterial from 'vue-material'
import VueCookies from 'vue-cookies'

import 'vue-material/dist/vue-material.min.css'
import 'vue-material/dist/theme/default.css'

export const bus = new Vue();

Vue.config.productionTip = false
export const ServerAddress = window.location.protocol + '//' + window.location.hostname + ':5000'
export const SocketInstance = io(ServerAddress, { secure: true })
// export const SocketInstance = io()

Vue.use(VueMaterial)

Vue.use(VueCookies)

Vue.use(new VueSocketIO({
  debug: true,
  connection: SocketInstance
}))

/* eslint-disable no-new */
new Vue({
  router,
  store,
  render: h => h(App),
  components: { App }
}).$mount('#app')
