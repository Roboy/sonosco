import Vue from 'vue'
import Router from 'vue-router'
import Transcriptor from '@/components/Transcriptor'
import ModelSelector from '@/components/ModelSelector'

Vue.use(Router)

export default new Router({
  routes: [
    {
      path: '/',
      name: 'Transcriptor',
      component: Transcriptor
    },
    {
      path: '/add',
      name: 'ModelSelector',
      component: ModelSelector
    }
  ]
})
